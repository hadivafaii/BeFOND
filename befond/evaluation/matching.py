"""Cosine matching for inexpensive progress checks and final evaluation."""

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import min_weight_full_bipartite_matching
from torch.nn.functional import normalize


MAX_SIMILARITY_ELEMENTS = 8_388_608
EXACT_MATCHING_MAX_LATENTS = 32_768


def final_matching_method(n_latents):
	return 'hungarian' if n_latents <= EXACT_MATCHING_MAX_LATENTS else 'top256'


@torch.no_grad()
def match_dictionary_atoms(decoder, truth, method='hungarian', *, absolute=False):
	"""Return CPU learned indices, truth indices, and their cosine scores.

	Hungarian and Top-k produce one-to-one matches. Max cosine allows reuse:
	learned-to-truth for absolute-cosine MCC, truth-to-learned for positive GT
	recovery. Top-k solves over the union of both directions' candidates plus
	diagonal edges, which guarantee a full rectangular matching is feasible.
	"""
	if method not in ('hungarian', 'max_cosine', 'top32', 'top256'):
		raise ValueError(f"Unknown dictionary matching method: {method!r}")
	decoder = normalize(decoder.detach().float(), dim=1)
	truth = normalize(truth.to(decoder).detach(), dim=1)
	reverse = method == 'max_cosine' and not absolute
	if reverse:
		decoder, truth = truth, decoder
	n, m = len(decoder), len(truth)

	def score(values):
		return values.abs_() if absolute else values.clamp_(0.0, 1.0)

	if method == 'hungarian':
		similarities = score(decoder @ truth.T).cpu().numpy()
		rows, columns = linear_sum_assignment(similarities, maximize=True)
		return (torch.from_numpy(rows), torch.from_numpy(columns),
			torch.from_numpy(similarities[rows, columns]).double())

	top_k = 256 if method == 'top256' else 32
	edges, values = [], []
	column_values = decoder.new_empty((0, m))
	column_rows = torch.empty((0, m), dtype=torch.long, device=decoder.device)
	block_size = max(1, MAX_SIMILARITY_ELEMENTS // m)
	for start in range(0, n, block_size):
		similarities = score(decoder[start:start + block_size] @ truth.T)
		rows = torch.arange(start, start + len(similarities), device=decoder.device)
		if method == 'max_cosine':
			best_values, columns = similarities.max(dim=1)
			edges.append(columns.cpu())
			values.append(best_values.cpu())
			continue
		row_values, columns = similarities.topk(min(top_k, m), dim=1)
		edges.append((rows[:, None] * m + columns).flatten().cpu().numpy())
		values.append(row_values.flatten().cpu().numpy())
		block_values, block_rows = similarities.topk(min(top_k, len(rows)), dim=0)
		all_values = torch.cat((column_values, block_values))
		all_rows = torch.cat((column_rows, block_rows + start))
		column_values, selected = all_values.topk(min(top_k, len(all_values)), dim=0)
		column_rows = all_rows.gather(0, selected)

	if method == 'max_cosine':
		rows, columns = torch.arange(n), torch.cat(edges)
		if reverse:
			rows, columns = columns, rows
		return rows, columns, torch.cat(values).double()

	columns = torch.arange(m, device=decoder.device)
	edges.append((column_rows * m + columns).flatten().cpu().numpy())
	values.append(column_values.flatten().cpu().numpy())
	diagonal = torch.arange(min(n, m), device=decoder.device)
	edges.append((diagonal * (m + 1)).cpu().numpy())
	values.append(score((decoder[:len(diagonal)] * truth[:len(diagonal)]).sum(1)).cpu().numpy())
	edges, first = np.unique(np.concatenate(edges), return_index=True)
	# Positive costs retain zero-similarity edges in scipy's sparse solver.
	costs = 2.0 - np.concatenate(values)[first].astype(np.float64)
	graph = csr_matrix((costs, (edges // m, edges % m)), shape=(n, m))
	rows, columns = min_weight_full_bipartite_matching(graph)
	cosines = torch.from_numpy(2.0 - np.asarray(graph[rows, columns]).ravel())
	return torch.from_numpy(rows), torch.from_numpy(columns), cosines
