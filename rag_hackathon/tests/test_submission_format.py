import ast

import pandas as pd

from rag_hack.pipeline import dataframe_to_submission


def test_submission_has_five_ids() -> None:
    answers = pd.DataFrame(
        {
            "q_id": [1, 2],
            "web_list": [[1, 2, 3, 4, 5], [10, 11, 12, 13, 14]],
        }
    )
    submission = dataframe_to_submission(answers)
    assert list(submission.columns) == ["q_id", "web_list"]
    for row in submission.itertuples():
        ids = ast.literal_eval(row.web_list)
        assert len(ids) == 5
        assert all(isinstance(x, int) for x in ids)
        assert len(set(ids)) == 5
