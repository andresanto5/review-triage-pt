"""Synthetic example against the local API, without paid services."""
import json
from urllib.request import Request, urlopen


def main():
    review = {'title': 'Produto com defeito', 'text': 'O produto chegou quebrado e não funciona.'}
    request = Request('http://127.0.0.1:8000/predict', data=json.dumps(review).encode('utf-8'),
                      headers={'Content-Type': 'application/json'})
    with urlopen(request, timeout=15) as response:
        result = json.load(response)
    assert result['review_required'] == (result['negative_score'] >= result['threshold'])
    assert result['score_calibrated'] is False
    print(json.dumps({'synthetic_example': review, 'response': result}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
