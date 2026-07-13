from src.med_circuitbench.sctc.circuits import EdgeCandidate, accept_edges, empirical_p_value


def test_empirical_p_value_and_edge_acceptance():
    p = empirical_p_value(0.5, [0.1, 0.2, 0.6])
    assert p == 0.5
    edges = [
        EdgeCandidate(0, 1, 1, 2, 0.8, 0.2, 0.001),
        EdgeCandidate(0, 2, 1, 3, 0.8, 0.05, 0.001),
        EdgeCandidate(0, 3, 1, 4, 0.8, 0.2, 0.9),
    ]
    accepted = accept_edges(edges)
    assert accepted[0].accepted
    assert not accepted[1].accepted
    assert not accepted[2].accepted
