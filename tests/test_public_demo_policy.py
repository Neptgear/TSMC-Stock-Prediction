import unittest
from public_demo_policy import public_parameters


class PublicDemoPolicyTests(unittest.TestCase):
    def test_bounds_advanced_inputs(self):
        data = public_parameters(dict(start='2023-01-01', end='2025-01-01',
                                      do='train_tft', d_model='99999', walkforward_splits='999'))
        self.assertEqual(data['d_model'], '32')
        self.assertEqual(data['walkforward_splits'], '1')
        self.assertEqual(data['use_fundamentals'], '0')

    def test_rejects_unbounded_or_unsafe_inputs(self):
        for values in [dict(do='train_both'), dict(horizon='30'), dict(epochs='200'),
                       dict(ticker='other'), dict(run_id='../secret'), dict(start='2010-01-01'),
                       dict(start='2025-01-01', end='2025-03-01')]:
            with self.subTest(values=values), self.assertRaises(ValueError):
                public_parameters(values)

    def test_health_and_invalid_requests_never_start_training(self):
        from app import create_app
        client = create_app(public_demo=True).test_client()
        headers = {'Host': 'localhost:5050'}
        self.assertTrue(client.get('/health', headers=headers).json['public_demo'])
        self.assertEqual(client.get('/?epochs=99999', headers=headers).status_code, 400)
        self.assertEqual(client.get('/?do=train_both', headers=headers).status_code, 400)
        self.assertEqual(client.get('/health', headers={'Host': 'evil.example'}).status_code, 403)
        self.assertEqual(client.get('/?do=train_tft', headers={**headers, 'Sec-Fetch-Site': 'cross-site'}).status_code, 403)
