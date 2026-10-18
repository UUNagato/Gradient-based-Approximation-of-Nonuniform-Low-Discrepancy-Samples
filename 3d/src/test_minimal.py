"""Small correctness and end-to-end tests for the standalone 3D release."""

from __future__ import annotations

import math
import argparse
import tempfile
import unittest
import json
import ast
import contextlib
import io
from unittest.mock import patch
from pathlib import Path

import numpy as np
import torch

from envmap import EnvironmentMap
from optimizer import (
    OptimizationConfig,
    batched_loss_and_backward,
    optimize_pointset,
    reference_loss,
)
from sphere import latent_to_direction, sample_spherical_caps, uniform_sample_sphere, cap_indicator
from targets import make_target, estimate_cap_masses, CosineTarget
import train_neural as neural
from train_neural import (
    AutoDecoderPointsetsV2cmp,
    ConditionalTarget,
    batched_loss_and_backward as neural_batched_loss_and_backward,
    estimate_cap_masses as estimate_conditional_cap_masses,
)
from visualize import load_pointset, render_pointset


THIS_DIR = Path(__file__).resolve().parent.parent
TEST_ENVMAP = THIS_DIR / "data" / "targets" / "example_envmap.npy"
TEST_EARTH = THIS_DIR / "data" / "targets" / "earth_map.png"


class Minimal3DTests(unittest.TestCase):
    def test_sphere_mapping_has_unit_length(self) -> None:
        latents = torch.tensor(
            [[0.0, 0.0], [0.2, 0.7], [0.5, 0.5], [1.2, -0.1]],
            dtype=torch.float32,
        )
        directions = latent_to_direction(latents)
        torch.testing.assert_close(
            torch.linalg.vector_norm(directions, dim=1),
            torch.ones(4),
            rtol=1e-5,
            atol=1e-5,
        )

    def test_batched_matches_reference_loss_and_gradient(self) -> None:
        device = torch.device("cpu")
        generator = torch.Generator(device=device).manual_seed(11)
        base_latents = torch.rand((13, 2), generator=generator, dtype=torch.float64)
        caps = sample_spherical_caps(
            17,
            device=device,
            dtype=torch.float64,
            generator=generator,
        )
        cap_masses = torch.rand((17,), generator=generator, dtype=torch.float64)

        reference_latents = base_latents.clone().requires_grad_(True)
        expected_loss = reference_loss(
            reference_latents,
            caps,
            cap_masses,
            tau=37.0,
        )
        expected_loss.backward()
        expected_gradient = reference_latents.grad.detach().clone()

        batched_latents = base_latents.clone().requires_grad_(True)
        actual_loss = batched_loss_and_backward(
            batched_latents,
            caps,
            cap_masses,
            tau=37.0,
            point_chunk_size=5,
            cap_chunk_size=6,
        )
        torch.testing.assert_close(actual_loss, expected_loss.detach(), rtol=1e-12, atol=1e-12)
        torch.testing.assert_close(
            batched_latents.grad,
            expected_gradient,
            rtol=1e-10,
            atol=1e-12,
        )

    def test_envmap_sampler_returns_valid_pdf(self) -> None:
        environment = EnvironmentMap(
            TEST_ENVMAP,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )
        generator = torch.Generator(device="cpu").manual_seed(4)
        directions, pdf = environment.sample(256, generator=generator)
        self.assertTrue(torch.isfinite(directions).all())
        self.assertTrue(torch.isfinite(pdf).all())
        self.assertTrue(torch.all(pdf > 0))
        torch.testing.assert_close(
            torch.linalg.vector_norm(directions, dim=1),
            torch.ones(256),
            rtol=1e-5,
            atol=1e-5,
        )

        uniform_directions = uniform_sample_sphere(
            torch.rand((65536, 2), generator=generator)
        )
        pdf_integral = environment.pdf(uniform_directions).mean() * (4.0 * math.pi)
        self.assertAlmostEqual(float(pdf_integral), 1.0, delta=0.05)

        image_environment = EnvironmentMap(
            TEST_EARTH,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )
        image_directions, image_pdf = image_environment.sample(64, generator=generator)
        self.assertEqual(image_directions.shape, (64, 3))
        self.assertTrue(torch.isfinite(image_pdf).all())

    def test_three_case_smoke_and_visualization(self) -> None:
        device = torch.device("cpu")
        config = OptimizationConfig(
            npoints=12,
            epochs=2,
            nregions=24,
            target_samples=128,
            target_sample_chunk_size=64,
            min_cap_hits=2,
            fallback_samples=8,
            point_chunk_size=5,
            cap_chunk_size=7,
            backend="batched",
            seed=3,
            snapshot_every=1,
        )
        results = {}
        for case in ("uniform", "envmap", "cosine"):
            target = make_target(
                case,
                device=device,
                dtype=torch.float32,
                envmap_path=TEST_ENVMAP if case == "envmap" else None,
            )
            result = optimize_pointset(target, config, device=device, progress=False)
            self.assertEqual(result.latents.shape, (12, 2))
            self.assertEqual(result.directions.shape, (12, 3))
            self.assertTrue(np.isfinite(result.losses).all())
            self.assertEqual(result.snapshot_epochs, [0, 1, 2])
            self.assertEqual(result.latent_snapshots.shape, (3, 12, 2))
            results[case] = result

        with tempfile.TemporaryDirectory() as temp_dir:
            pointset_path = Path(temp_dir) / "pointset.npz"
            image_path = Path(temp_dir) / "pointset.png"
            np.savez_compressed(
                pointset_path,
                latents=results["uniform"].latents.numpy(),
                directions=results["uniform"].directions.numpy(),
            )
            loaded = load_pointset(pointset_path)
            render_pointset(loaded, image_path, image_size=360)
            self.assertEqual(loaded.shape, (12, 3))
            self.assertTrue(image_path.is_file())
            self.assertGreater(image_path.stat().st_size, 0)

    def test_neural_brdf_and_env_backward(self) -> None:
        device = torch.device("cpu")
        generator = torch.Generator(device=device).manual_seed(17)
        common = dict(
            envmap=TEST_ENVMAP,
            view_angle=35.0,
            roughness=0.35,
            metallic=0.2,
            anisotropic=0.0,
            subsurface=0.0,
            normal_condition=True,
            view_condition=True,
            offset_condition=True,
            point_chunk_size=6,
            cap_chunk_size=8,
        )
        base_uv = torch.quasirandom.SobolEngine(2, scramble=True, seed=17).draw(12)
        normal = torch.tensor([0.2, 0.3, 0.9327379])
        offset = torch.tensor([0.17, 0.43])
        caps = sample_spherical_caps(
            16, device=device, dtype=torch.float32, generator=generator
        )
        space_directions = uniform_sample_sphere(torch.rand((128, 2), generator=generator))

        for case in ("brdf", "env"):
            args = argparse.Namespace(case=case, **common)
            target = ConditionalTarget(args, device)
            cap_masses = estimate_conditional_cap_masses(
                target, caps, space_directions, normal, args.cap_chunk_size
            )
            model = AutoDecoderPointsetsV2cmp(neural.condition_dim(args), [24, 24, 24, 24], 2, 12, .1, 16)
            loss = neural_batched_loss_and_backward(
                model, base_uv, normal, offset, caps, cap_masses,
                tau=30.0, gradient_scale=1.0, args=args,
            )
            gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
            self.assertTrue(torch.isfinite(loss))
            self.assertTrue(gradients)
            self.assertTrue(all(torch.isfinite(gradient).all() for gradient in gradients))
            self.assertGreater(sum(float(gradient.abs().sum()) for gradient in gradients), 0.0)

    def test_exact_cap_membership_and_finite_endpoint_gradients(self):
        caps = torch.tensor([[0., 0., 1., .005], [0., 0., 1., math.pi-.005]], dtype=torch.float64)
        points = torch.tensor([[0., 0., 1.], [0., 0., -1.]], dtype=torch.float64, requires_grad=True)
        membership = cap_indicator(caps, points, 37.)
        torch.testing.assert_close(membership, torch.tensor([[1., 1.], [0., 0.]], dtype=torch.float64))
        membership.sum().backward()
        self.assertTrue(torch.isfinite(points.grad).all())

    def test_mirror_integer_boundaries_are_continuous(self):
        for axis in (0, 1):
            for integer in range(-2, 3):
                points = torch.full((3, 2), .2, dtype=torch.float64)
                points[:, axis] = torch.tensor([integer-1e-7, integer, integer+1e-7], dtype=torch.float64)
                directions = latent_to_direction(points)
                torch.testing.assert_close(directions[0], directions[1], atol=1e-6, rtol=0)
                torch.testing.assert_close(directions[2], directions[1], atol=1e-6, rtol=0)

    def test_white_envmap_is_uniform_in_solid_angle(self):
        generator = torch.Generator().manual_seed(23)
        for shape in ((1, 1), (2, 4), (17, 33)):
            env = EnvironmentMap(np.ones(shape), device=torch.device('cpu'), dtype=torch.float64)
            directions, pdf = env.sample(65536, generator=generator)
            torch.testing.assert_close(pdf, torch.full_like(pdf, 1/(4*math.pi)), atol=1e-12, rtol=1e-12)
            torch.testing.assert_close(env.pdf(directions), pdf, atol=1e-12, rtol=1e-12)
            poles = torch.tensor([[0., 0., 1.], [0., 0., -1.]], dtype=torch.float64)
            torch.testing.assert_close(env.pdf(poles), torch.full((2,), 1/(4*math.pi), dtype=torch.float64))
            self.assertAlmostEqual(float(directions[:, 2].square().mean()), 1/3, delta=.005)

    def test_envmap_zero_texels_and_uniform_endpoints(self):
        image = np.zeros((128, 256), dtype=np.float32)
        image[40, 20] = 3
        env = EnvironmentMap(image, device=torch.device('cpu'))
        uniforms = torch.tensor([[0., 0., 0.], [1., 1., 1.], [.5, .5, .5]], dtype=torch.float64)
        directions, pdf = env.sample_from_uniform(uniforms)
        self.assertTrue(torch.all(pdf > 0))
        torch.testing.assert_close(env.pdf(directions), pdf)
        self.assertTrue(torch.isfinite(directions).all())

    def test_envmap_texel_frequencies_match_luminance_times_area(self):
        image = np.zeros((8, 16), dtype=np.float64)
        image[0, 1], image[5, 13] = 7., 2.
        env = EnvironmentMap(image, device=torch.device('cpu'), dtype=torch.float64)
        directions, pdf = env.sample(65536, generator=torch.Generator().manual_seed(12))
        area_top = (2*math.pi/16) * (1-math.cos(math.pi/8))
        area_lower = (2*math.pi/16) * (math.cos(5*math.pi/8)-math.cos(6*math.pi/8))
        probability = 7*area_top/(7*area_top+2*area_lower)
        top = directions[:, 2] > math.cos(math.pi/8)
        self.assertAlmostEqual(float(top.double().mean()), probability, delta=.005)
        torch.testing.assert_close(env.pdf(directions), pdf)
        self.assertAlmostEqual(float(pdf[top][0]), 7/(7*area_top+2*area_lower))

    def test_independent_fallback_has_correct_expectation(self):
        class UniformProposal:
            def density(self, directions):
                return torch.full(directions.shape[:-1], 1/(4*math.pi), dtype=directions.dtype)
            def sample_proposal(self, count, *, generator):
                directions = uniform_sample_sphere(torch.rand((count, 2), dtype=torch.float64, generator=generator))
                return directions, self.density(directions)
        generator = torch.Generator().manual_seed(2026)
        probability = .0625
        caps = torch.tensor([[0., 0., 1., math.acos(1-2*probability)]], dtype=torch.float64)
        values = [float(estimate_cap_masses(
            UniformProposal(), caps, sample_count=128, cap_chunk_size=1,
            sample_chunk_size=128, min_cap_hits=8, fallback_samples=8, generator=generator
        )[0]) for _ in range(2000)]
        # The previous shared-gate estimator returns about .071, well outside this bound.
        self.assertAlmostEqual(float(np.mean(values)), probability, delta=.002)

    def test_probability_estimates_are_not_clipped(self):
        target = CosineTarget(device=torch.device('cpu'), dtype=torch.float64)
        directions = torch.tensor([[0., 0., 1.]], dtype=torch.float64).expand(8, -1)
        pdf = torch.full((8,), 1/(2*math.pi), dtype=torch.float64)
        caps = torch.tensor([[0., 0., 1., math.pi]], dtype=torch.float64)
        with patch.object(target, 'sample_proposal', return_value=(directions, pdf)):
            masses = estimate_cap_masses(target, caps, sample_count=8, cap_chunk_size=1,
                                        sample_chunk_size=8, min_cap_hits=1, fallback_samples=8,
                                        generator=torch.Generator())
        self.assertAlmostEqual(float(masses[0]), 2.)

    def test_brdf_conditions_are_local_views(self):
        args = neural.build_parser().parse_args(['--case', 'brdf'])
        neural.validate_args(args)
        self.assertEqual(neural.condition_dim(args), 4)
        target = ConditionalTarget(args, torch.device('cpu'))
        generator = torch.Generator().manual_seed(5)
        views = torch.stack([neural.sample_condition(args, generator, torch.device('cpu')) for _ in range(32)])
        self.assertTrue(torch.all(views[:, 2] > 0))
        self.assertGreater(float(views[:, 2].std()), .15)
        wi = torch.tensor([[0., 0., 1.], [.6, 0., .8], [0., 0., -1.]])
        results = []
        for wo in (views[0], views[1]):
            expected = neural.disney_brdf(wi, wo.expand_as(wi), roughness=args.roughness,
                metallic=args.metallic, anisotropic=args.anisotropic,
                subsurface=args.subsurface) * wi[:, 2].clamp_min(0)
            actual = target.weights(wi, wo)
            torch.testing.assert_close(actual, expected)
            self.assertEqual(float(actual[-1]), 0.)
            results.append(actual)
        self.assertFalse(torch.allclose(*results))
        args.fixed_normal = [0, 0, 1]
        with self.assertRaisesRegex(ValueError, 'local normal'):
            neural.validate_args(args)

    def test_ggx_mixture_matches_known_rough_limit(self):
        args = neural.build_parser().parse_args(['--case', 'brdf', '--roughness', '1'])
        target = ConditionalTarget(args, torch.device('cpu'))
        wo = torch.tensor([0., 0., 1.], dtype=torch.float64)
        directions, pdf = target.sample_proposal(65536, wo, generator=torch.Generator().manual_seed(8))
        # At alpha=1 and normal view, reflected GGX NDF is uniform on the sphere.
        expected = .5/(4*math.pi) + .5 * directions[:, 2].clamp_min(0)/math.pi
        torch.testing.assert_close(pdf, expected, atol=1e-10, rtol=1e-10)
        self.assertAlmostEqual(float(directions[:, 2].mean()), 1/3, delta=.005)
        self.assertAlmostEqual(float(directions[:, 2].square().mean()), 5/12, delta=.005)

    def test_importance_sampling_handles_sparse_env_and_sharp_brdf(self):
        generator = torch.Generator().manual_seed(9)
        condition = torch.tensor([0., 0., 1.])
        with tempfile.TemporaryDirectory() as td:
            path = Path(td)/'sparse.npy'
            image = np.zeros((128, 256), dtype=np.float32)
            image[40, 20] = 1
            np.save(path, image)
            args = neural.build_parser().parse_args(['--case', 'env', '--envmap', str(path)])
            target = ConditionalTarget(args, torch.device('cpu'))
            caps = torch.tensor([[0., 0., 1., math.pi/2]])
            for _ in range(8):
                directions, pdf = target.sample_proposal(4096, condition, generator=generator)
                masses = estimate_conditional_cap_masses(target, caps, directions, condition, 1, pdf)
                torch.testing.assert_close(masses, torch.ones_like(masses))
                self.assertGreater(target.last_effective_sample_size, 1900)
        args = neural.build_parser().parse_args(['--case', 'brdf', '--roughness', '.01', '--metallic', '1'])
        target = ConditionalTarget(args, torch.device('cpu'))
        directions, pdf = target.sample_proposal(8192, condition, generator=generator)
        caps = torch.tensor([[0., 0., 1., .02], [0., 0., -1., math.pi/2]])
        masses = estimate_conditional_cap_masses(target, caps, directions, condition, 1, pdf)
        self.assertGreater(float(masses[0]), .95)
        self.assertEqual(float(masses[1]), 0.)
        self.assertGreater(target.last_effective_sample_size, 3000)

    def test_anisotropic_brdf_mixture_agrees_with_independent_quadrature(self):
        args = neural.build_parser().parse_args(['--case', 'brdf', '--roughness', '.55', '--anisotropic', '.65'])
        target = ConditionalTarget(args, torch.device('cpu'))
        condition = torch.tensor([.3, .4, math.sqrt(.75)], dtype=torch.float64)
        uniform = torch.quasirandom.SobolEngine(2, scramble=True, seed=14).draw(262144).double()
        reference_dirs = uniform_sample_sphere(uniform)
        reference_weights = target.weights(reference_dirs, condition)
        expected = (reference_weights[:, None]*reference_dirs).sum(0)/reference_weights.sum()
        directions, pdf = target.sample_proposal(65536, condition, generator=torch.Generator().manual_seed(13))
        weights = target.weights(directions, condition)/pdf
        actual = (weights[:, None]*directions).sum(0)/weights.sum()
        torch.testing.assert_close(actual, expected, atol=.012, rtol=0)

    def test_fixed_brdf_view_and_ablation_options(self):
        args = neural.build_parser().parse_args(['--case', 'brdf', '--fixed-view', '3', '0', '4',
                                                '--no-view-condition', '--no-offset-condition'])
        neural.validate_args(args)
        self.assertEqual(neural.condition_dim(args), 0)
        condition = neural.sample_condition(args, torch.Generator(), torch.device('cpu'))
        torch.testing.assert_close(condition, torch.tensor([.6, 0., .8]))
        model = AutoDecoderPointsetsV2cmp(0, [16, 16, 16, 16], 2, 4, .1, 16)
        base = torch.rand((4, 2))
        points = neural.model_directions(model, base, condition, torch.zeros(2), slice(None), args)
        self.assertTrue(torch.isfinite(points).all())
        conditions, _, _ = neural.evaluate_conditions(model, base, args, torch.device('cpu'))
        np.testing.assert_allclose(conditions, np.tile([.6, 0., .8], (3, 1)), atol=1e-7)

    def test_original_decoder_structure_and_zero_residual_mapping(self):
        args = neural.build_parser().parse_args(['--case', 'brdf'])
        model = AutoDecoderPointsetsV2cmp(4, args.hidden_sizes, 2, 13, .1, 16)
        self.assertFalse(hasattr(model, 'codebook'))
        self.assertEqual(model.network[-1].linear.out_features, 26)
        self.assertFalse(model.encoding_layer.B.requires_grad)
        model.initialize_residual()
        base = torch.rand(13, 2)
        condition = torch.tensor([.3, .4, math.sqrt(.75)])
        offset = torch.tensor([.83, .67])
        points = neural.model_directions(model, base, condition, offset, slice(None), args)
        torch.testing.assert_close(points, latent_to_direction(base + offset))

    def test_release_example_fixes_specular_to_zero(self):
        args = neural.build_parser().parse_args(['--case', 'brdf'])
        self.assertFalse(hasattr(args, 'specular'))
        for value in ('0', '0.5'):
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                neural.build_parser().parse_args(['--case', 'brdf', '--specular', value])
        neural.validate_args(args)
        args.specular = 0.5
        with self.assertRaisesRegex(ValueError, 'specular to 0'):
            neural.validate_args(args)

    def test_quadrature_and_shared_sobol_have_correct_solid_angle_measure(self):
        class ConstantTarget:
            def weights(self, points, condition):
                return (points[:, 2] > 0).to(points.dtype)
        target = ConstantTarget()
        quadrature = neural.BRDFQuadrature(torch.device('cpu'), 16)
        view = torch.tensor([0., 0., 1.])
        self.assertAlmostEqual(float(quadrature.integral(target, view)), 2 * math.pi, places=12)
        caps = torch.tensor([[0., 0., 1., math.acos(.5)], [0., 0., 1., math.pi]])
        masses = neural.shared_sobol_masses(target, caps, [view, view], sample_count=4096,
                                          seed=3, chunk_size=1, quadrature=quadrature)
        torch.testing.assert_close(masses, torch.tensor([[.5, .5], [1., 1.]]), atol=1e-6, rtol=0)

    def test_neural_resume_matches_uninterrupted_training(self):
        with tempfile.TemporaryDirectory() as directory:
            args = neural.build_parser().parse_args(['--case', 'brdf', '--epochs', '4',
                '--npoints', '8', '--hidden-sizes', '8', '8', '--condition-batch-size', '2',
                '--nregions', '16', '--target-samples', '64', '--normalization-resolution', '8',
                '--validation-every', '0', '--output-dir', directory])
            torch.manual_seed(31)
            full, _, base, losses = neural.train(args, torch.device('cpu'))
            torch.manual_seed(31)
            args.stop_after = 2
            neural.train(args, torch.device('cpu'))
            args.stop_after = None
            args.resume = Path(directory) / 'training_state.pt'
            resumed, _, resumed_base, resumed_losses = neural.train(args, torch.device('cpu'))
            self.assertEqual(losses, resumed_losses)
            torch.testing.assert_close(base, resumed_base, atol=0, rtol=0)
            for name, value in full.state_dict().items():
                torch.testing.assert_close(value, resumed.state_dict()[name], atol=0, rtol=0)

    def test_ggx_near_antipodal_reflection_keeps_positive_pdf(self):
        args = neural.build_parser().parse_args(['--case', 'brdf'])
        target = ConditionalTarget(args, torch.device('cpu'))
        condition = torch.tensor([-.7873794436454773, -.6142359972000122, .05241912603378296])
        ax, _ = target._ggx_parameters()
        slope = -condition[2].double() / condition[0].double() / ax
        u = slope.square() / (1 + slope.square())
        values = torch.tensor([[.5, .5], [float(u) + 1e-10, 0.]], dtype=torch.float64)
        with patch.object(torch, 'rand', side_effect=lambda *a, **kw: values.to(kw['dtype'])):
            points, pdf = target.sample_proposal(2, condition, generator=torch.Generator())
        self.assertEqual(points.dtype, torch.float64)
        self.assertTrue(torch.isfinite(pdf).all())
        self.assertTrue((pdf > 0).all())
        masses = estimate_conditional_cap_masses(target, torch.tensor([[0., 0., 1., math.pi/2]]),
                                                points, condition, 1, pdf)
        self.assertTrue(torch.isfinite(masses).all())

    def test_neural_parameter_gradients_match_direct_composition(self):
        torch.manual_seed(31)
        for case in ('brdf', 'env'):
            args = neural.build_parser().parse_args(['--case', case, '--point-chunk-size', '5', '--cap-chunk-size', '6'])
            model = AutoDecoderPointsetsV2cmp(neural.condition_dim(args), [24, 24, 24, 24], 2, 13, .1, 16).double()
            with torch.no_grad():
                model.network[-1].linear.weight.normal_(0, .02)
            base = torch.rand((13, 2), dtype=torch.float64)
            condition = torch.tensor([.2, .3, .9327379053], dtype=torch.float64)
            offset = torch.tensor([.17, .43], dtype=torch.float64)
            caps = sample_spherical_caps(17, device=torch.device('cpu'), dtype=torch.float64,
                                         generator=torch.Generator().manual_seed(19))
            masses = torch.rand(17, dtype=torch.float64)
            points = neural.model_directions(model, base, condition, offset, slice(None), args)
            expected_loss = torch.sqrt((cap_indicator(caps, points, 37).mean(0)-masses).square().mean()+1e-12)
            (expected_loss/3).backward()
            expected_grads = [p.grad.clone() for p in model.parameters() if p.requires_grad]
            model.zero_grad(set_to_none=True)
            actual_loss = neural_batched_loss_and_backward(model, base, condition, offset, caps, masses, 37, 1/3, args)
            torch.testing.assert_close(actual_loss, expected_loss.detach(), atol=1e-12, rtol=1e-12)
            for parameter, expected in zip((p for p in model.parameters() if p.requires_grad), expected_grads):
                torch.testing.assert_close(parameter.grad, expected, atol=1e-11, rtol=1e-9)

    def test_notebook_geometry_and_envmap_regressions(self):
        notebook = json.loads((THIS_DIR/'notebooks'/'earth_fast_forward.ipynb').read_text(encoding='utf-8'))
        namespace = dict(torch=torch, np=np, math=math, Path=Path,
                         DEVICE=torch.device('cpu'), DTYPE=torch.float64)
        exec(''.join(notebook['cells'][4]['source']), namespace)
        tree = ast.parse(''.join(notebook['cells'][6]['source']))
        class_node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'EnvironmentMap')
        exec(compile(ast.Module(body=[class_node], type_ignores=[]), '<notebook-env>', 'exec'), namespace)
        env = namespace['EnvironmentMap'](np.ones((2, 4)), device=torch.device('cpu'), dtype=torch.float64)
        directions, pdf = env.sample(4096, generator=torch.Generator().manual_seed(2))
        torch.testing.assert_close(pdf, torch.full_like(pdf, 1/(4*math.pi)))
        caps = torch.tensor([[0., 0., 1., .005]], dtype=torch.float64)
        points = torch.tensor([[0., 0., 1.]], dtype=torch.float64)
        self.assertEqual(float(namespace['cap_indicator'](caps, points, 20)[0, 0]), 1.)


if __name__ == "__main__":
    unittest.main()
