import sys
sys.path.insert(0, 'src')
from ground_truth import load_ground_truth, load_ground_truth_pages, validate_pages_against_papers

gt = load_ground_truth()
gtp = load_ground_truth_pages()
problems = validate_pages_against_papers(gt, gtp)
print(problems if problems else 'OK: files match')