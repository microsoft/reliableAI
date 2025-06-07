from src.run import *
from itertools import combinations

def read_mapping(map_json_path) -> Dict[str, Dict[int, str]]:
    with open(map_json_path, "r") as f:
        map_json = json.load(f)
    return map_json

def gen_explanation(
    dm: XplainerDataModel,
    fp: FixedPredicate, 
    dp: DiffPredicate,
    measure: Measure, 
    prior_col: List[str],
    map_json: Dict[str, List[str]]
) -> Dict[str, List[str] | str]:
    result = {
        "whyquery": {
            "diff_predicate": {
                "col": dp.col,
                "vals": [map_json[dp.col][str(v)] for v in dp.vals]
            },
            "measure": {
                "col": measure.col,
                "func": measure.func
            },
            "explanation_col": prior_col
        },
        "explanation": []
    }

    dq = AvgDiffQuery(fp, dp, measure, dm.col_names)
    dq.create_scope(dm.conn)
    logging.info(f"Diff predicate {dp.col} {dp.vals[0]} {dp.vals[1]}: {dq.get_raw_diff(dm.conn)}")
    if dq.get_raw_diff(dm.conn) < 0:
        dq = AvgDiffQuery(fp, DiffPredicate(dp.col, list(reversed(dp.vals))), measure, dm.col_names)
        dq.create_scope(dm.conn)
        logging.info(f"Reversed diff predicate {dp.col} {dp.vals[0]} {dp.vals[1]}: {dq.get_raw_diff(dm.conn)}")
        assert dq.get_raw_diff(dm.conn) > 0
    search = AvgSearch(dq, dm.value_set, dm.conn)

    exp_cols = prior_col
    
    search.run(exp_cols)
    for exp, resp in search.explanation:
        result["explanation"].append({
            "counterfactual_predicate": {
                "col": exp.col,
                "vals": [map_json[exp.col][str(v)] for v in exp.vals]
            },
            "responsibility": resp
        })
    return result

if __name__ == "__main__":
# Define and parse program input arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", help="Path to data file", required=True)  
    parser.add_argument("--query", help="Path to query JSON file", required=True) 
    parser.add_argument("--map", help="Path to map JSON file", required=True)
    args = parser.parse_args()

    data_path = args.data
    query_json_path = args.query
    map_json_path = args.map

    logging.basicConfig(level=logging.INFO)

    logging.info(f"Reading data from {data_path}")
    dm = XplainerDataModel(data_path)

    logging.info(f"Reading map from {map_json_path}")
    map_json = read_mapping(map_json_path)

    logging.info(f"Reading diff query from {query_json_path}")
    fp, dp, measure, prior_col = read_diff_query(query_json_path)
    
    all_result = []

    logging.info(f"Enumerating diff query")

    for comb in combinations(map_json[dp.col].keys(), 2):
        dp = DiffPredicate(dp.col, comb)
        result = gen_explanation(dm, fp, dp, measure, prior_col, map_json)
        all_result.append(result)
        logging.info(f"Done {comb}")
        logging.info(f"Result: {result}")
    
    logging.info(f"Writing result to {query_json_path}.result")
    with open("result.json", "w") as f:
        json.dump(all_result, f, indent=4)