import torch


def get_constr_out(x, R):
    """
    Given the network output x and a constraint matrix R,
    returns the modified output according to the hierarchical constraints in R.
    """
    # Convert x to double precision
    c_out = x.double()

    # Add a dimension to c_out: from (N, D) to (N, 1, D)
    # N: batch size, D: dimensionality of the output
    c_out = c_out.unsqueeze(1)

    # Expand c_out to match the shape of R:
    # If R is (C, C), c_out becomes (N, C, C)
    c_out = c_out.expand(len(x), R.shape[1], R.shape[1])

    # Expand R similarly to (N, C, C)
    R_batch = R.expand(len(x), R.shape[1], R.shape[1])

    # Element-wise multiplication of R_batch by c_out.
    # This produces a (N, C, C) tensor.
    # torch.max(...) is taken along dimension=2, resulting in (N, C).
    # This extracts the maximum along the last dimension,
    # effectively applying the hierarchical constraints.
    final_out, _ = torch.max(R_batch * c_out.double(), dim=2)

    return final_out
