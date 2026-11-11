
from src.processing.clean import clean_text, clean_records


def test_clean_text():

    raw_text = """
    
    
    Machine Learning     improves     data analysis.


    Supervised Learning uses labeled data.
    
    
    """

    result = clean_text(raw_text)

    assert result == (
        "Machine Learning improves data analysis.\n\n"
        "Supervised Learning uses labeled data."
    )

    print("\nText cleaning successful!")
    print(result)


def test_clean_records():

    records = [
        {
            "course_name": "Statistics",
            "page": 1,
            "text": "  Normal    distribution.  "
        },
        {
            "course_name": "Economics",
            "page": 2,
            "text": "   "
        },
        {
            "course_name": "Finance",
            "page": 3,
            "text": "Present value = FV / (1 + r)^n"
        }
    ]

    results = clean_records(records)

    assert len(results) == 2

    assert results[0]["text"] == "Normal distribution."

    assert results[1]["text"] == (
        "Present value = FV / (1 + r)^n"
    )

    assert results[0]["course_name"] == "Statistics"
    assert results[0]["page"] == 1

    print("\nRecord cleaning successful!")
    print(results)