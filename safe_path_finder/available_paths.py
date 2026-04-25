"""
start->
KOL_001
KOL_002
KOL_003


end->
KGP_008
KGP_009
KGP_010
"""

"""
combinations:
KOL_001 - KGP_008
KOL_001 - KGP_009
KOL_001 - KGP_010

KOL_002 - KGP_008
KOL_002 - KGP_009
KOL_002 - KGP_010

KOL_003 - KGP_008
KOL_003 - KGP_009
KOL_003 - KGP_010
"""

"""
{
KOL_001 - KGP_008: {
    paths: [
        ["KOL_001", "KOL_002", "KOL_005", "KOL_007", "KGP_008"],
        ["KOL_001", "KOL_004", "KOL_006", "KGP_008"],
        ["KOL_001", "KOL_002", "KOL_006", "KOL_007", "KGP_008"],
        ["KOL_001", "KOL_003", "KOL_005", "KGP_008"],
        ["KOL_001", "KOL_002", "KOL_007", "KGP_008"],
    ]
},
KOL_001 - KGP_009: {
    paths: [
        ["KOL_001", "KOL_002", "KOL_005", "KGP_009"],
        ["KOL_001", "KOL_004", "KOL_006", "KGP_009"],
        ["KOL_001", "KOL_002", "KOL_006", "KOL_007", "KGP_009"],
        ["KOL_001", "KOL_003", "KOL_005", "KGP_009"],
        ["KOL_001", "KOL_002", "KOL_007", "KGP_009"],
    ]
},
KOL_001 - KGP_010: {
    paths: [
        ["KOL_001", "KOL_002", "KOL_005", "KGP_009", "KGP_010"],
        ["KOL_001", "KOL_004", "KOL_006", "KGP_010"],
        ["KOL_001", "KOL_002", "KOL_006", "KOL_007", "KGP_010"],
        ["KOL_001", "KOL_003", "KOL_005", "KGP_010"],
        ["KOL_001", "KOL_002", "KGP_009", "KGP_010"],
    ]
},

KOL_002 - KGP_008: {
    paths: [
        ["KOL_002", "KOL_005", "KOL_007", "KGP_008"],
        ["KOL_002", "KOL_004", "KOL_006", "KGP_008"],
        ["KOL_002", "KOL_006", "KOL_007", "KGP_008"],
        ["KOL_002", "KOL_003", "KOL_005", "KOL_007", "KGP_008"],
        ["KOL_002", "KOL_003", "KOL_007", "KGP_008"],
    ]
},
KOL_002 - KGP_009: {
    paths: [
        ["KOL_002", "KOL_005", "KOL_007", "KGP_008", "KGP_009"],
        ["KOL_002", "KOL_004", "KOL_006", "KGP_009"],
        ["KOL_002", "KOL_006", "KOL_007", "KGP_008", "KGP_009"],
        ["KOL_002", "KOL_003", "KOL_005", "KOL_007", "KGP_008", "KGP_009"],
        ["KOL_002", "KOL_003", "KGP_009"],
    ]
},
KOL_002 - KGP_010: {
    paths: [
        ["KOL_002", "KOL_005", "KOL_007", "KGP_008", "KGP_010"],
        ["KOL_002", "KOL_004", "KOL_006", "KGP_009", "KGP_010"],
        ["KOL_002", "KOL_006", "KOL_007", "KGP_008", "KGP_010"],
        ["KOL_002", "KOL_003", "KOL_005", "KOL_007", "KGP_010"],
        ["KOL_002", "KOL_003", "KOL_007", "KGP_008", "KGP_010"],
    ]
},
KOL_003 - KGP_008: {
    paths: [
        ["KOL_003", "KOL_005", "KOL_007", "KGP_008"],
        ["KOL_003", "KOL_004", "KOL_006", "KGP_008"],
        ["KOL_003", "KOL_006", "KOL_007", "KGP_008"],
        ["KOL_003", "KOL_005", "KOL_007", "KGP_008"],
        ["KOL_003", "KOL_007", "KGP_008"],
    ]
},
KOL_003 - KGP_009: {
    paths: [
        ["KOL_003", "KOL_005", "KOL_007", "KGP_009"],
        ["KOL_003", "KOL_004", "KOL_006", "KGP_008", "KGP_009"],
        ["KOL_003", "KOL_006", "KOL_007", "KGP_009"],
        ["KOL_003", "KOL_005", "KOL_007", "KGP_008", "KGP_009"],
        ["KOL_003", "KOL_007", "KGP_008", "KGP_009"],
    ]
},
KOL_003 - KGP_010: {
    paths: [
        ["KOL_003", "KOL_005", "KOL_007", "KGP_008", "KGP_010"],
        ["KOL_003", "KOL_004", "KOL_006", "KGP_009", "KGP_010"],
        [KOL_003", "KOL_006", "KOL_007", "KGP_008", "KGP_010"],
        ["KOL_003", "KOL_005", "KOL_007", "KGP_008", , "KGP_009", "KGP_010"],
        ["KOL_003", "KOL_007", "KGP_008", "KGP_010"],
    ]
}
}

"""
available_paths = {
    "KOL_001 - KGP_008": {
        "paths": [
            ["KOL_001", "KOL_002", "KOL_005", "KOL_007", "KGP_008"],
            ["KOL_001", "KOL_004", "KOL_006", "KGP_008"],
            ["KOL_001", "KOL_002", "KOL_006", "KOL_007", "KGP_008"],
            ["KOL_001", "KOL_003", "KOL_005", "KGP_008"],
            ["KOL_001", "KOL_002", "KOL_007", "KGP_008"],
        ]
    },
    "KOL_001 - KGP_009": {
        "paths": [
            ["KOL_001", "KOL_002", "KOL_005", "KGP_009"],
            ["KOL_001", "KOL_004", "KOL_006", "KGP_009"],
            ["KOL_001", "KOL_002", "KOL_006", "KOL_007", "KGP_009"],
            ["KOL_001", "KOL_003", "KOL_005", "KGP_009"],
            ["KOL_001", "KOL_002", "KOL_007", "KGP_009"],
        ]
    },
    "KOL_001 - KGP_010": {
        "paths": [
            ["KOL_001", "KOL_002", "KOL_005", "KGP_009", "KGP_010"],
            ["KOL_001", "KOL_004", "KOL_006", "KGP_010"],
            ["KOL_001", "KOL_002", "KOL_006", "KOL_007", "KGP_010"],
            ["KOL_001", "KOL_003", "KOL_005", "KGP_010"],
            ["KOL_001", "KOL_002", "KGP_009", "KGP_010"],
        ]
    },
    "KOL_002 - KGP_008": {
        "paths": [
            ["KOL_002", "KOL_005", "KOL_007", "KGP_008"],
            ["KOL_002", "KOL_004", "KOL_006", "KGP_008"],
            ["KOL_002", "KOL_006", "KOL_007", "KGP_008"],
            ["KOL_002", "KOL_003", "KOL_005", "KOL_007", "KGP_008"],
            ["KOL_002", "KOL_003", "KOL_007", "KGP_008"],
        ]
    },
    "KOL_002 - KGP_009": {
        "paths": [
            ["KOL_002", "KOL_005", "KOL_007", "KGP_008", "KGP_009"],
            ["KOL_002", "KOL_004", "KOL_006", "KGP_009"],
            ["KOL_002", "KOL_006", "KOL_007", "KGP_008", "KGP_009"],
            ["KOL_002", "KOL_003", "KOL_005", "KOL_007", "KGP_008", "KGP_009"],
            ["KOL_002", "KOL_003", "KGP_009"],
        ]
    },
    "KOL_002 - KGP_010": {
        "paths": [
            ["KOL_002", "KOL_005", "KOL_007", "KGP_008", "KGP_010"],
            ["KOL_002", "KOL_004", "KOL_006", "KGP_009", "KGP_010"],
            ["KOL_002", "KOL_006", "KOL_007", "KGP_008", "KGP_010"],
            ["KOL_002", "KOL_003", "KOL_005", "KOL_007", "KGP_010"],
            ["KOL_002", "KOL_003", "KOL_007", "KGP_008", "KGP_010"],
        ]
    },
    "KOL_003 - KGP_008": {
        "paths": [
            ["KOL_003", "KOL_005", "KOL_007", "KGP_008"],
            ["KOL_003", "KOL_004", "KOL_006", "KGP_008"],
            ["KOL_003", "KOL_006", "KOL_007", "KGP_008"],
            ["KOL_003", "KOL_005", "KOL_007", "KGP_008"],
            ["KOL_003", "KOL_007", "KGP_008"],
        ]
    },
    "KOL_003 - KGP_009": {
        "paths": [
            ["KOL_003", "KOL_005", "KOL_007", "KGP_009"],
            ["KOL_003", "KOL_004", "KOL_006", "KGP_008", "KGP_009"],
            ["KOL_003", "KOL_006", "KOL_007", "KGP_009"],
            ["KOL_003", "KOL_005", "KOL_007", "KGP_008", "KGP_009"],
            ["KOL_003", "KOL_007", "KGP_008", "KGP_009"],
        ]
    },
    "KOL_003 - KGP_010": {
        "paths": [
            ["KOL_003", "KOL_005", "KOL_007", "KGP_008", "KGP_010"],
            ["KOL_003", "KOL_004", "KOL_006", "KGP_009", "KGP_010"],
            ["KOL_003", "KOL_006", "KOL_007", "KGP_008", "KGP_010"],
            ["KOL_003", "KOL_005", "KOL_007", "KGP_008", "KGP_009", "KGP_010"],
            ["KOL_003", "KOL_007", "KGP_008", "KGP_010"],
        ]
    },
}
