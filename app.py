from flask import Flask
from flask import request
from flask import render_template

from settings import team_name_mappings

from tools import team_schedule_table
from tools import team_vs_team_table, test_heuristic, get_team_schedule, test_heuristic_range
from tools import heuristic_basic, heuristic_linear_regression, heuristic_decision_tree, heuristic_kth_nearest
from tools import heuristic_combined_ai, heuristic_random
from tools import heuristic_perceptron  # added here because of MatLab dependency

app = Flask(__name__)

heuristic_mapping = {
    "h_random": heuristic_random,
    "h_basic": heuristic_basic,
    "h_linear": heuristic_linear_regression,
    "h_dtree": heuristic_decision_tree,
    "h_kth": heuristic_kth_nearest,
    "h_combined": heuristic_combined_ai,
    "h_perceptron": heuristic_perceptron,
}


@app.route("/", methods=['GET', 'POST'])
def hello():
    return render_template("home.html")


@app.route("/compare/")
def get_offense():
    values = request.values
    team1 = values["Team1"]
    year1 = int(values["Year1"])
    team2 = values["Team2"]
    year2 = int(values["Year2"])
    h1 = heuristic_basic(team1, year1, team2, year2, False, False, True)
    h2 = heuristic_linear_regression(team1, year1, team2, year2, False, False, True)
    h3 = heuristic_decision_tree(team1, year1, team2, year2, False, False, True)
    h4 = heuristic_kth_nearest(team1, year1, team2, year2, False, False, True)
    h5 = heuristic_combined_ai(team1, year1, team2, year2, False, False, True)
    team1_full = team_name_mappings[team1]["full"]
    team2_full = team_name_mappings[team2]["full"]
    return team_vs_team_table(team1_full, year1, team2_full, year2, h1, h2, h3, h4, h5)


@app.route("/season/")
def get_season():
    values = request.values
    team = values["Team"]
    year = int(values["Year"])
    heuristic = heuristic_mapping[values["Heuristic"]]
    result = get_team_schedule(team, year, heuristic=heuristic)
    return team_schedule_table(result)


@app.route("/accuracy/")
def get_accuracy():
    values = request.values
    heuristic = heuristic_mapping[values["Heuristic"]]
    year = int(values["Year"])
    result = str(test_heuristic(heuristic, year, False))
    return "<p>%s</p><p>Accuracy: %s</p>" % (str(year), result)


@app.route("/total_accuracy/")
def get_total_accuracy():
    values = request.values
    heuristic = heuristic_mapping[values["Heuristic"]]
    years = [2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017]
    result = str(test_heuristic_range(heuristic, years, False))
    return "<p>2010-2017</p><p>Accuracy: %s</p>" % result


if __name__ == '__main__':
    app.run(debug=True)