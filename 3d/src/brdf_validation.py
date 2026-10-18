"""Independent normalization and held-out checks for BRDF training."""
import math
import numpy as np
import torch
from sphere import uniform_sample_sphere, sample_spherical_caps


class BRDFQuadrature:
    """Integrate over the hemisphere in z and azimuth, including anisotropy.

    Unlike a polar-only LUT, this evaluates the actual outgoing direction.
    Nodes are fixed, independent of the cap integration samples.
    """
    def __init__(self, device, resolution=128):
        z, weights = np.polynomial.legendre.leggauss(resolution)
        z = torch.tensor((z + 1) / 2, device=device, dtype=torch.float64)
        weights = torch.tensor(weights / 2, device=device, dtype=torch.float64)
        phi = (torch.arange(2 * resolution, device=device, dtype=torch.float64) + .5) * math.pi / resolution
        zz, pp = torch.meshgrid(z, phi, indexing='ij')
        radius = torch.sqrt(1 - zz.square())
        self.directions = torch.stack((radius * pp.cos(), radius * pp.sin(), zz), -1).reshape(-1, 3)
        self.weights = (weights[:, None].expand(-1, 2 * resolution) * math.pi / resolution).reshape(-1)

    @torch.no_grad()
    def integral(self, target, condition):
        value = (target.weights(self.directions, condition) * self.weights).sum()
        if not torch.isfinite(value) or value <= 0:
            raise FloatingPointError('Invalid BRDF normalization integral.')
        return value


@torch.no_grad()
def shared_sobol_masses(target, caps, conditions, *, sample_count, seed, chunk_size, quadrature):
    """One scrambled Sobol sphere sample set, shared by all view conditions."""
    uv = torch.quasirandom.SobolEngine(2, scramble=True, seed=seed).draw(sample_count, dtype=torch.float64).to(caps.device)
    directions = uniform_sample_sphere(uv)
    columns = [target.weights(directions, condition) * (4 * math.pi / sample_count)
               / quadrature.integral(target, condition) for condition in conditions]
    weights = torch.stack(columns, 1)
    target.last_effective_sample_size = float((weights[:, -1].sum().square() / weights[:, -1].square().sum()).cpu())
    result = torch.empty((len(caps), len(conditions)), device=caps.device, dtype=caps.dtype)
    geometry = directions.to(caps.dtype)
    # Shared membership is the expensive part; reuse it across conditions.
    for start in range(0, len(caps), chunk_size):
        block = caps[start:start + chunk_size]
        inside = geometry @ block[:, :3].T >= block[:, 3].cos()[None]
        result[start:start + chunk_size] = inside.to(weights.dtype).T @ weights
    return result


class HeldOutValidation:
    def __init__(self, target, device, quadrature):
        self.conditions = torch.tensor([
            [0., 0., 1.], [2**-.5, 0., 2**-.5], [0., 2**-.5, 2**-.5],
            [math.sin(math.radians(75)), 0., math.cos(math.radians(75))],
            [0., math.sin(math.radians(75)), math.cos(math.radians(75))],
        ], device=device)
        self.offsets = torch.tensor([[0., 0.], [.37, .19], [.11, .61]], device=device)
        self.caps = sample_spherical_caps(512, device=device, dtype=torch.float32,
                                         generator=torch.Generator(device=device).manual_seed(982451653))
        self.references = shared_sobol_masses(target, self.caps, self.conditions,
            sample_count=262144, seed=961748941, chunk_size=128, quadrature=quadrature)

    @torch.no_grad()
    def evaluate(self, model, base, args):
        from train_neural import model_directions
        values, below = [], []
        for index, condition in enumerate(self.conditions):
            for offset in self.offsets:
                points = model_directions(model, base, condition, offset, slice(None), args)
                mass = (points @ self.caps[:, :3].T >= self.caps[:, 3].cos()[None]).float().mean(0)
                values.append(float((mass - self.references[:, index]).square().mean().sqrt()))
                below.append(int((points[:, 2] < 0).sum()))
        return {'mean_cap_l2': float(np.mean(values)), 'cap_l2': values, 'below_hemisphere': below}
