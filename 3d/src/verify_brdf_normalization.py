"""Check quadrature convergence before a BRDF neural training run."""
import json
import math
import torch
from train_neural import build_parser, validate_args, resolve_device, ConditionalTarget
from brdf_validation import BRDFQuadrature
from sphere import uniform_sample_sphere


def main():
    args = build_parser().parse_args()
    validate_args(args)
    if args.case != 'brdf' or args.output_dir is None:
        raise ValueError('Use --case brdf and --output-dir.')
    device = resolve_device(args.device)
    target = ConditionalTarget(args, device)
    quadratures = [BRDFQuadrature(device, args.normalization_resolution * factor) for factor in (1, 2)]
    samples = uniform_sample_sphere(torch.quasirandom.SobolEngine(2, scramble=True, seed=49979687)
                                   .draw(1048576, dtype=torch.float64).to(device))
    records = []
    with torch.no_grad():
        for theta in (0, 30, 45, 60, 75, 85, 89):
            for phi in (0, 45, 90):
                t, p = math.radians(theta), math.radians(phi)
                view = torch.tensor([math.sin(t)*math.cos(p), math.sin(t)*math.sin(p), math.cos(t)],
                                    device=device, dtype=torch.float64)
                values = [float(q.integral(target, view)) for q in quadratures]
                qmc = float(target.weights(samples, view).mean() * 4 * math.pi)
                records.append(dict(theta=theta, phi=phi, integral=values[0], refined_integral=values[1],
                                    relative_refinement_error=abs(values[0]/values[1]-1),
                                    independent_qmc_integral=qmc, relative_qmc_difference=abs(qmc/values[1]-1)))
    report = dict(material=dict(roughness=args.roughness, metallic=args.metallic,
                               fixed_specular=0., anisotropic=args.anisotropic, subsurface=args.subsurface),
                  resolution=args.normalization_resolution, qmc_samples=len(samples), views=records)
    report['max_relative_refinement_error'] = max(r['relative_refinement_error'] for r in records)
    report['max_relative_qmc_difference'] = max(r['relative_qmc_difference'] for r in records)
    report['passed'] = report['max_relative_refinement_error'] < 1e-4 and report['max_relative_qmc_difference'] < 1e-3
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / 'normalization_check.json').write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != 'views'}, indent=2), flush=True)
    if not report['passed']:
        raise RuntimeError('Normalization check failed; increase resolution or investigate before training.')


if __name__ == '__main__':
    main()
