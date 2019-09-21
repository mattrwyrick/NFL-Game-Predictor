import os


PROJ_DIR = os.path.dirname(os.path.realpath(__file__))

team_name_mappings = {
    "ARI":	{"full": "Arizona Cardinals", "name1": "Arizona"},
    "ATL":	{"full": "Atlanta Falcons", "name1": "Atlanta"},
    "BAL":	{"full": "Baltimore Ravens", "name1": "Baltimore"},
    "BUF":	{"full": "Buffalo Bills", "name1": "Buffalo"},
    "CAR":	{"full": "Carolina Panthers", "name1": "Carolina"},
    "CHI":	{"full": "Chicago Bears", "name1": "Chicago"},
    "CIN":	{"full": "Cincinnati Bengals", "name1": "Cincinnati"},
    "CLE":	{"full": "Cleveland Browns", "name1": "Cleveland"},
    "DAL":	{"full": "Dallas Cowboys", "name1": "Dallas"},
    "DEN":	{"full": "Denver Broncos", "name1": "Denver"},
    "DET":	{"full": "Detroit Lions", "name1": "Detroit"},
    "GB":	{"full": "Green Bay Packers", "name1": "Green Bay"},
    "HOU":	{"full": "Houston Texans", "name1": "Houston"},
    "IND":	{"full": "Indianapolis Colts", "name1": "Indianapolis"},
    "JAX":	{"full": "Jacksonville Jaguars", "name1": "Jacksonville"},
    "KC":	{"full": "Kansas City Chiefs", "name1": "Kansas City"},
    "LAC":	{"full": "Los Angeles Chargers", "name1": "LA Chargers"},
    "LAR":	{"full": "Los Angeles Rams", "name1": "LA Rams"},
    "SLR":	{"full": "St. Louis Rams", "name1": "LA Rams"},  # account for 2016 Rams city move
    "MIA":	{"full": "Miami Dolphins", "name1": "Miami"},
    "MIN":	{"full": "Minnesota Vikings", "name1": "Minnesota"},
    "NE":	{"full": "New England Patriots", "name1": "New England"},
    "NO":	{"full": "New Orleans Saints", "name1": "New Orleans"},
    "NYG":	{"full": "New York Giants", "name1": "NY Giants"},
    "NYJ":	{"full": "New York Jets", "name1": "NY Jets"},
    "OAK":	{"full": "Oakland Raiders", "name1": "Oakland"},
    "PHI":	{"full": "Philadelphia Eagles", "name1": "Philadelphia"},
    "PIT":	{"full": "Pittsburgh Steelers", "name1": "Pittsburgh"},
    "SEA":	{"full": "Seattle Seahawks", "name1": "Seattle"},
    "SF":	{"full": "San Francisco 49ers", "name1": "San Francisco"},
    "TB":	{"full": "Tampa Bay Buccaneers", "name1": "Tampa Bay"},
    "TEN":	{"full": "Tennessee Titans", "name1": "Tennessee"},
    "WAS":	{"full": "Washington Redskins", "name1": "Washington"}
}

team_name_mappings2 = {
    "Arizona Cardinals": "ARI",
    "Atlanta Falcons": "ATL",
    "Baltimore Ravens": "BAL",
    "Buffalo Bills": "BUF",
    "Carolina Panthers": "CAR",
    "Chicago Bears": "CHI",
    "Cincinnati Bengals": "CIN",
    "Cleveland Browns": "CLE",
    "Dallas Cowboys": "DAL",
    "Denver Broncos": "DEN",
    "Detroit Lions": "DET",
    "Green Bay Packers": "GB",
    "Houston Texans": "HOU",
    "Indianapolis Colts": "IND",
    "Jacksonville Jaguars": "JAX",
    "Kansas City Chiefs": "KC",
    "Los Angeles Chargers": "LAC",
    "San Diego Chargers": "LAC",
    "Los Angeles Rams": "LAR",
    "St. Louis Rams": "SLR",
    "Miami Dolphins": "MIA",
    "Minnesota Vikings": "MIN",
    "New England Patriots": "NE",
    "New Orleans Saints": "NO",
    "New York Giants": "NYG",
    "New York Jets": "NYJ",
    "Oakland Raiders": "OAK",
    "Philadelphia Eagles": "PHI",
    "Pittsburgh Steelers": "PIT",
    "Seattle Seahawks": "SEA",
    "San Francisco 49ers": "SF",
    "Tampa Bay Buccaneers": "TB",
    "Tennessee Titans": "TEN",
    "Washington Redskins":  "WAS",
}

training_years = [2016, 2014, 2012]
