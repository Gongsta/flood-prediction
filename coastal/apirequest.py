import cdsapi

c = cdsapi.Client()

c.retrieve(
    'satellite-sea-level-global',
    {
        'year': '1995',
        'month': '01',
        'day': [
            '01', '02', '03',
        ],
        'variable': 'all',
        'format': 'zip',
    },
    'coastsample.zip')