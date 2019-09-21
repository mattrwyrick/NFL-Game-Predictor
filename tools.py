import csv
import os
import math
import random

from settings import PROJ_DIR, team_name_mappings, team_name_mappings2, training_years


def print_out_dictionary(data):
    output = ""
    for key in data:
        output += key.rjust(20) + "&emsp;&emsp;" + str(data[key]) + "<br>"
    return output


def team_vs_team_table(team1, year1, team2, year2, h1, h2, h3, h4, h5):
    winner = team1 + " " + str(year1) if h5 == "Win" else team2 + " " + str(year2)
    html = '<link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/4.1.3/css/bootstrap.min.css">'
    html += '<br><h1>%s %s vs %s %s</h1>' % (team1, str(year1), team2, str(year2))
    html += "<h3>Winner: %s</h3><br>" % winner
    html += "<p>Table from Perspective of Home Team"
    html += '<table class="table"><thead class="thead-dark">'
    html += "<tr><td>%s</td><td>%s</td><td>%s</td><td>%s</td></tr>" % ("Basic", "Linear Regression", "Decision Tree", "Kth Nearest")
    html += "<tbody>"
    html += "<tr><td>%s</td><td>%s</td><td>%s</td><td>%s</td></tr>" % (h1, h2, h3, h4)
    html += "</tbody></table>"
    return html


def get_offensive_stat(team, year, stat, post_season=False):
    file_pathway = get_offense_file_pathway(year, post_season)
    with open(file_pathway, newline='') as data:
        reader = csv.DictReader(data)
        for row in reader:
            if row["TEAM"] == team:
                return row[stat]
        return None


def get_team_stats(team, year, ps=False):
    team_pts = float(get_offensive_stat(team_name_mappings[team]["name1"], year, "PTS", ps))
    team_ptsg = float(get_offensive_stat(team_name_mappings[team]["name1"], year, "PTS/G", ps))
    team_yds = float(get_offensive_stat(team_name_mappings[team]["name1"], year, "YDS", ps))
    team_ydsg = float(get_offensive_stat(team_name_mappings[team]["name1"], year, "YDS/G", ps))
    team_pass = float(get_offensive_stat(team_name_mappings[team]["name1"], year, "PASS", ps))
    team_passg = float(get_offensive_stat(team_name_mappings[team]["name1"], year, "P YDS/G", ps))
    team_rush = float(get_offensive_stat(team_name_mappings[team]["name1"], year, "RUSH", ps))
    team_rushg = float(get_offensive_stat(team_name_mappings[team]["name1"], year, "R YDS/G", ps))

    stats = {"PTS": team_pts, "PTS/G": team_ptsg, "YDS": team_yds, "YDS/G": team_ydsg, "PASS": team_pass,
            "P YDS/G": team_passg, "RUSH": team_rush, "R YDS/G": team_rushg}

    return stats


def get_offensive_stat_difference(stat, year1, team1, year2, team2, ps1=False, ps2=False):
    stat1 = get_offensive_stat(year1, team1, stat, ps1)
    stat2 = get_offensive_stat(year2, team2, stat, ps2)
    return stat1 - stat2


def get_offensive_stat_sum(stat, year1, team1, year2, team2, ps1=False, ps2=False):
    stat1 = get_offensive_stat(year1, team1, stat, ps1)
    stat2 = get_offensive_stat(year2, team2, stat, ps2)
    return stat1 + stat2


def get_file_path(*paths):
    pathway = PROJ_DIR
    for item in paths:
        pathway = os.path.join(pathway, item)
    return pathway


def get_offense_file_pathway(year, post_season):
    season_dir = "postSeason" if post_season else "regSeason"
    file_prefix = "post" if post_season else "regular"
    file_name = file_prefix + str(year) + ".csv"
    file_pathway = get_file_path("data", "offensive", season_dir, file_name)
    return file_pathway


def get_offensive_team_data(team, year, post_season=False):
    file_pathway = get_offense_file_pathway(year, post_season)
    with open(file_pathway, newline='') as data:
        reader = csv.DictReader(data)
        for row in reader:
            if row["TEAM"] == team:
                return {"Team": team, "YDS": row["YDS"], "YDS/G": row["YDS/G"], "PASS": row["PASS"],
                        "P YDS/G": row["P YDS/G"], "RUSH": row["RUSH"], "R YDS/G": row["R YDS/G"],
                        "PTS": row["PTS"], "PTS/G": row["PTS/G"]}
    return {}


def get_weekly_file_pathway(year, post_season):
    file_prefix = "post" if post_season else "regular"
    file_name = file_prefix + str(year) + ".csv"
    file_pathway = get_file_path("data", "weekly", file_name)
    return file_pathway


def get_team_schedule(team, year, post_season=False, heuristic=lambda a, b, c, d, e, f, g: "N/A"):
    season = {"team": team, "year": str(year), "post": post_season, "outcome": []}
    outcome = season["outcome"]
    file_pathway = get_weekly_file_pathway(year, post_season)
    if year < 2016 and team == "LAR":
        team_full = team_name_mappings["SLR"]["full"]  # Rams old city
    else:
        team_full = team_name_mappings[team]["full"]

    with open(file_pathway, newline='') as data:
        reader = csv.DictReader(data)
        for row in reader:

            if row["Winner/tie"] == team_full or row["Loser/tie"] == team_full:
                    win = "Win" if row["Winner/tie"] == team_full else "Loss"
                    home = (win and row["visit"] != "@") or ((not win) and row["visit"] == "@")
                    opponent = row["Loser/tie"] if win == "Win" else row["Winner/tie"]
                    if row["PtsW"] == "":
                        win = "N/A"

                    t1 = team_name_mappings2[team_full]
                    t2 = team_name_mappings2[opponent]
                    prediction = heuristic(t1, year, t2, year, post_season, post_season, home)

                    game = {"win": win, "prediction": prediction, "home": home, "opponent": opponent}
                    outcome.append(game)

        season["accuracy"] = season_accuracy(season)
        return season


def team_schedule_table(season):
    html = '<link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/4.1.3/css/bootstrap.min.css">'
    html += '<h1>%s %s</h1>' % (season["team"], season["year"])
    html += "<h2>Post Season</h2>" if season["post"] else "<h2>Regular Season</h2>"
    html += "<h3>Accuracy: %s" % str(season["accuracy"])
    html += '<table class="table"><thead class="thead-dark">'
    html += _make_schedule_row("RESULT", "PREDICTION", "HOME", "OPPONENT")
    html += "<tbody>"
    for game in season["outcome"]:
        symbol = "" if game["home"] else "@"
        html += _make_schedule_row(game["win"], game["prediction"], symbol, game["opponent"])
    html += "</tbody></table>"
    return html


def _make_schedule_row(actual, prediction, home, opponent):
    return "<tr><td>%s</td><td>%s</td><td>%s</td><td>%s</td>" % (actual, prediction, home, opponent)


def heuristic_random(team1, year1, team2, year2, ps1=False, ps2=False, home1=False):
    result = "Win" if random.randint(0,1) % 2 == 0 else "Loss"
    return result


def heuristic_combined_ai(team1, year1, team2, year2, ps1=False, ps2=False, home1=False):
    result = dtreeCombined.heuristic(team1, year1, team2, year2, ps1, ps2, home1)
    return result


def heuristic_decision_tree(team1, year1, team2, year2, ps1=False, ps2=False, home1=False):
    result = dtree.heuristic(team1, year1, team2, year2, ps1, ps2, home1)
    return result


def heuristic_kth_nearest(team1, year1, team2, year2, ps1=False, ps2=False, home1=False):
    result = k_near.heuristic(team1, year1, team2, year2, ps1, ps2, home1)
    return result


def heuristic_linear_regression(team1, year1, team2, year2, ps1=False, ps2=False, home1=False):
    x = 4
    team1_p1 = float(get_offensive_stat(team_name_mappings[team1]["name1"], year1 - 1, "PTS/G", ps1))
    team1_p2 = float(get_offensive_stat(team_name_mappings[team1]["name1"], year1 - 2, "PTS/G", ps1))
    team1_px = float(get_offensive_stat(team_name_mappings[team1]["name1"], year1 - x, "PTS/G", ps1))

    team2_p1 = float(get_offensive_stat(team_name_mappings[team2]["name1"], year2 - 1, "PTS/G", ps2))
    team2_p2 = float(get_offensive_stat(team_name_mappings[team2]["name1"], year2 - 2, "PTS/G", ps2))
    team2_px = float(get_offensive_stat(team_name_mappings[team2]["name1"], year2 - x, "PTS/G", ps2))

    y1 = [team1_p1, team1_p2, team1_px]
    x1 = [year1-1, year1-2, year1-x]
    y2 = [team2_p1, team2_p2, team2_px]
    x2 = [year2-1, year2-2, year2-x]

    team1_lr = linear_regression_estimate(y1, x1, year1)
    team2_lr = linear_regression_estimate(y2, x2, year2)

    if team2_lr > team1_lr:
        return "Loss"

    return "Win"


def linear_regression_estimate(ys, xs, year):
    y_ave = sum(ys) / len(ys)
    x_ave = sum(xs) / len(xs)

    rise = 0
    run = 0
    for y in ys:
        for x in xs:
            rise += (x - x_ave) * (y - y_ave)
            run += (x - x_ave)**2

    m = rise / run
    b = y_ave - (m * x_ave)

    estimate = (m * year) + b

    return estimate


# def heuristic_kth_nearest(team1, year1, team2, year2, ps1=False, ps2=False, home1=False)
#     result = kth_nearest.heuristic(team1, year1, team2, year2, ps1, ps2, home1)
#     return result


def heuristic_basic(team1, year1, team2, year2, ps1=False, ps2=False, home1=False):
    team1_stat1 = float(get_offensive_stat(team_name_mappings[team1]["name1"], year1-1, "YDS/G", ps1))
    team1_stat4 = float(get_offensive_stat(team_name_mappings[team1]["name1"], year1-1, "P YDS/G", ps1))
    team1_stat6 = float(get_offensive_stat(team_name_mappings[team1]["name1"], year1-1, "R YDS/G", ps1))
    team1_stat8 = float(get_offensive_stat(team_name_mappings[team1]["name1"], year1-1, "PTS/G", ps1))

    team2_stat1 = float(get_offensive_stat(team_name_mappings[team2]["name1"], year2-1, "YDS/G", ps2))
    team2_stat4 = float(get_offensive_stat(team_name_mappings[team2]["name1"], year2-1, "P YDS/G", ps2))
    team2_stat6 = float(get_offensive_stat(team_name_mappings[team2]["name1"], year2-1, "R YDS/G", ps2))
    team2_stat8 = float(get_offensive_stat(team_name_mappings[team2]["name1"], year2-1, "PTS/G", ps2))

    A = 13.7
    B = 1.1
    C = 1.1
    D = 3.5
    E1 = 1.1 if home1 else 1.0
    E2 = 1 if home1 else 1.1
    team1 = ((team1_stat8 * A) + (team1_stat1 * B) + (team1_stat4 * C) + (team1_stat6 * D)) * E1
    team2 = ((team2_stat8 * A) + (team2_stat1 * B) + (team2_stat4 * C) + (team2_stat6 * D)) * E2

    if (team1 - team2) < 0:
        return "Loss"
    return "Win"


def season_accuracy(season):
    total = 0
    correct = 0
    for game in season["outcome"]:
        if (game["win"] == "Win") or (game["win"] == "Loss"):
            total += 1
            if game["win"] == game["prediction"]:
                correct += 1

    if total > 0:
        return float(correct) / float(total)

    return 0


def test_heuristic(heuristic, year, post_season=False):
    total = 0
    accuracy = 0.0
    for key in team_name_mappings:
        total += 1
        team_season = get_team_schedule(key, year, post_season, heuristic)
        accuracy += team_season["accuracy"]

    if total == 0:
        return 0

    return accuracy / total


def test_heuristic_range(heuristic, years, post_season=False):
    total = 0
    accuracy = 0.0
    for year in years:
        accuracy += test_heuristic(heuristic, year, post_season)
        total += 1

    if total == 0:
        return 0

    return accuracy / total


#######################
#  DECISION TREE CODE #
#######################

def entropy(mi, m):
    if mi == 0:
        return 0
    return -1.0 * (mi / m) * math.log2(mi / m)


class TreeTrainerStats:

    def __init__(self):
        pass

    def train(self, years, ps=False):
        result = self.build_up_collection(years, ps)
        all_stats = result[0]
        all_games = result[1]
        stat_keys = ["PTS", "PTS/G", "YDS", "YDS/G", "PASS", "P YDS/G", "RUSH", "R YDS/G", "HOME"]
        result = self._recursive_train(stat_keys, all_stats, all_games)
        return result

    def _recursive_train(self, stat_keys, all_stats, all_games, outcome="N/A"):
        if not stat_keys:
            return self._calc_outcome(all_games)

        if not all_games:
            return outcome

        if self._completely_classifeid(all_games):
            return all_games[0][2]

        split_stat = self._calc_entropy(stat_keys, all_stats, all_games)
        if not split_stat:  # splits don't separate anymore
            return self._calc_outcome(all_games)

        left_bin = []
        right_bin = []

        for game in all_games:
            year, team1, win1, team2, home = game

            if split_stat == "HOME":
                if home:
                   left_bin.append(game)
                else:
                    right_bin.append(game)

            else:
                stat1 = all_stats[year][team1][split_stat]
                stat2 = all_stats[year][team2][split_stat]

                if stat2 > stat1:
                    left_bin.append(game)
                else:
                    right_bin.append(game)

        stat_keys.remove(split_stat)

        left_result = self._recursive_train(stat_keys.copy(), all_stats, left_bin)
        right_result = self._recursive_train(stat_keys.copy(), all_stats, right_bin)

        result = [split_stat, left_result, right_result]
        return result

    def _completely_classifeid(self, games):
        if not games:
            return True
        outcome = games[0][2]
        for game in games:
            if game[2] != outcome:
                return False
        return True

    def _calc_outcome(self, games):
        loss = 0
        win = 0
        for game in games:
            if game[2] == "Loss":
                loss += 1
            else:
                win += 1
        if loss > win:
            return "Loss"
        return "Win"

    def _calc_entropy(self, stat_keys, all_stats, all_games):
        h = []
        m = len(all_games)
        for key in stat_keys:
            mx = 0
            m1 = 0
            m1w = 0
            m2 = 0
            m2w = 0
            for game in all_games:
                year, team1, win1, team2, home = game
                if win1 == "Loss":
                    mx += 1

                if key == "HOME":  # home field advantage
                    if home:
                        m1 +=1
                        if win1 == "Loss":
                            m1w += 1
                    else:
                        m2 += 1
                        if win1 == "Loss":
                            m2w += 1

                else:  # regular stat
                    stat1 = all_stats[year][team1][key]
                    stat2 = all_stats[year][team2][key]
                    if stat2 > stat1:
                        m1 += 1
                        if win1 == "Loss":
                            m1w += 1
                    else:
                        m2 += 1
                        if win1 == "Loss":
                            m2w += 1

            e1 = entropy(mx, m)
            e2 = entropy(m - mx, m)
            e1i1 = entropy(m1w, m1)
            e1i2 = entropy(m1 - m1w, m1)
            e2i1 = entropy(m2w, m2)
            e2i2 = entropy(m2 - m2w, m2)
            i1 = (m1/m)*(e1i1 + e1i2)
            i2 = (m2/m)*(e2i1 + e2i2)
            e = (e1 + e2) - (i1 + i2)
            h.append([e, key])

        h.sort(reverse=True)
        if h[0][0] == 0:
            return None
        best_key = h[0][1]
        return best_key

    def build_up_collection(self, years, ps=False):
        all_stats = {}
        all_games = []

        for year in years:
            all_stats[year] = {}

        for year in years:
            for team1 in team_name_mappings:
                all_stats[year][team1] = self._get_team_stats(team1, year, ps)

                season = get_team_schedule(team1, year, post_season=ps)
                for game in season["outcome"]:
                    team2 = team_name_mappings2[game["opponent"]]
                    team1_win = game["win"]
                    home = game["home"]
                    result = [year, team1, team1_win, team2, home]
                    all_games.append(result)

        return all_stats, all_games

    def _get_team_stats(self, team, year, ps=False):
        team_pts = float(get_offensive_stat(team_name_mappings[team]["name1"], year, "PTS", ps))
        team_ptsg = float(get_offensive_stat(team_name_mappings[team]["name1"], year, "PTS/G", ps))
        team_yds = float(get_offensive_stat(team_name_mappings[team]["name1"], year, "YDS", ps))
        team_ydsg = float(get_offensive_stat(team_name_mappings[team]["name1"], year, "YDS/G", ps))
        team_pass = float(get_offensive_stat(team_name_mappings[team]["name1"], year, "PASS", ps))
        team_passg = float(get_offensive_stat(team_name_mappings[team]["name1"], year, "P YDS/G", ps))
        team_rush = float(get_offensive_stat(team_name_mappings[team]["name1"], year, "RUSH", ps))
        team_rushg = float(get_offensive_stat(team_name_mappings[team]["name1"], year, "R YDS/G", ps))

        stats = {"PTS": team_pts, "PTS/G": team_ptsg, "YDS": team_yds, "YDS/G": team_ydsg, "PASS": team_pass,
                 "P YDS/G": team_passg, "RUSH": team_rush, "R YDS/G": team_rushg}

        return stats


class DecisionTree:

    def __init__(self, years, post_season=False):
        trainer = TreeTrainerStats()
        decisions = trainer.train(years, post_season)
        self.decisions = decisions

    def heuristic(self, team1, year1, team2, year2, ps1=False, ps2=False, home1=True):
        team1_stat = self._get_team_stats(team1, year1, ps1)
        team2_stat = self._get_team_stats(team2, year2, ps2)
        team1_stat["HOME"] = home1
        team2_stat["HOME"] = not home1
        result = self.iter_tree(self.decisions, team1_stat, team2_stat)
        return result

    def iter_tree(self, dtree, team1_stat, team2_stat):
        if type(dtree) == str:
            return dtree

        stat = dtree[0]

        if stat == "HOME":
            if team1_stat["HOME"]:
                return self.iter_tree(dtree[1], team1_stat, team2_stat)
            else:
                return self.iter_tree(dtree[2], team1_stat, team2_stat)

        else:
            if team2_stat[stat] > team1_stat[stat]:
                return self.iter_tree(dtree[1], team1_stat, team2_stat)
            else:
                return self.iter_tree(dtree[2], team1_stat, team2_stat)

    def _get_team_stats(self, team, year, ps=False):
        team_pts = float(get_offensive_stat(team_name_mappings[team]["name1"], year, "PTS", ps))
        team_ptsg = float(get_offensive_stat(team_name_mappings[team]["name1"], year, "PTS/G", ps))
        team_yds = float(get_offensive_stat(team_name_mappings[team]["name1"], year, "YDS", ps))
        team_ydsg = float(get_offensive_stat(team_name_mappings[team]["name1"], year, "YDS/G", ps))
        team_pass = float(get_offensive_stat(team_name_mappings[team]["name1"], year, "PASS", ps))
        team_passg = float(get_offensive_stat(team_name_mappings[team]["name1"], year, "P YDS/G", ps))
        team_rush = float(get_offensive_stat(team_name_mappings[team]["name1"], year, "RUSH", ps))
        team_rushg = float(get_offensive_stat(team_name_mappings[team]["name1"], year, "R YDS/G", ps))

        stats = {"PTS": team_pts, "PTS/G": team_ptsg, "YDS": team_yds, "YDS/G": team_ydsg, "PASS": team_pass,
                 "P YDS/G": team_passg, "RUSH": team_rush, "R YDS/G": team_rushg}

        return stats


dtree = DecisionTree(training_years)


########################
# Kth Nearest Neighbor #
########################

class KVec:

    def __init__(self, s1, s2, s3, s4, s5, s6, s7, s8, classification=None):
        self.vector = [s1, s2, s3, s4, s5, s6, s7, s8]
        self.classification = classification

    def distance(self, other_kvec):
        sum_dist_2 = 0
        for i, v1 in enumerate(other_kvec.vector):
            v2 = self.vector[i]
            sum_dist_2 += (v1 - v2) ** 2
        return sum_dist_2 ** .5


class KthNearest:

    def __init__(self, k, years, post_season=False):
        self.k = k
        self.games = []
        for year in years:
            for team in team_name_mappings:
                stats1 = get_team_stats(team, year, post_season)
                season = get_team_schedule(team, year, post_season)
                for game in season["outcome"]:
                    stats2 = get_team_stats(team_name_mappings2[game["opponent"]], year, post_season)
                    classification = game["win"]

                    vector = KVec(stats1["PTS"]-stats2["PTS"], stats1["PTS/G"]-stats2["PTS/G"],
                                  stats1["YDS"]-stats2["YDS"], stats1["YDS/G"]-stats2["YDS/G"],
                                  stats1["PASS"]-stats2["PASS"], stats1["P YDS/G"]-stats2["P YDS/G"],
                                  stats1["RUSH"]-stats2["RUSH"], stats1["R YDS/G"]-stats2["R YDS/G"],
                                  classification)
                    self.games.append(vector)

    def heuristic(self, team1, year1, team2, year2, ps1=False, ps2=False, home1=True):
        stats1 = get_team_stats(team1, year1, ps1)
        stats2 = get_team_stats(team2, year2, ps2)
        vector = KVec(stats1["PTS"] - stats2["PTS"], stats1["PTS/G"] - stats2["PTS/G"],
                      stats1["YDS"] - stats2["YDS"], stats1["YDS/G"] - stats2["YDS/G"],
                      stats1["PASS"] - stats2["PASS"], stats1["P YDS/G"] - stats2["P YDS/G"],
                      stats1["RUSH"] - stats2["RUSH"], stats1["R YDS/G"] - stats2["R YDS/G"],
                      )

        distances = []
        for point in self.games:
            classification = point.classification
            distance = vector.distance(point)
            result = [distance, classification]
            distances.append(result)

        distances.sort()
        losses = 0
        for i in range(self.k):
            if distances[i][1] == "Loss":
                losses += 1

        if losses > self.k/2:
            return "Loss"

        return "Win"


k_near = KthNearest(100, training_years)


def get_heuristic_mapping():
    return {
        "h_basic": heuristic_basic,
        "h_linear": heuristic_linear_regression,
        "h_dtree": heuristic_decision_tree,
        "h_kth": heuristic_kth_nearest,
    }


class TreeTrainerCombinedAI:

    def __init__(self):
        self.ps = False

    def train(self, years, ps=False):
        self.ps = ps
        all_games = self.build_up_collection(years, ps)
        functions = get_heuristic_mapping()
        stat_keys = ["h_basic", "h_linear", "h_dtree", "h_kth"]
        result = self._recursive_train(stat_keys, functions, all_games)
        return result

    def _recursive_train(self, stat_keys, functions, all_games, outcome="N/A"):
        if not stat_keys:
            return self._calc_outcome(all_games)

        if not all_games:
            return outcome

        if self._completely_classifeid(all_games):
            return all_games[0][2]

        h_name = self._calc_entropy(stat_keys, functions, all_games)
        if not h_name:  # splits don't separate anymore
            return self._calc_outcome(all_games)

        left_bin = []
        right_bin = []

        for game in all_games:
            year, team1, win1, team2, home = game
            heuristic = functions[h_name]
            result = heuristic(team1, year, team2, year, self.ps, self.ps, home)

            if result != "Win":
                left_bin.append(game)
            else:
                right_bin.append(game)

        stat_keys.remove(h_name)

        left_result = self._recursive_train(stat_keys.copy(), functions, left_bin)
        right_result = self._recursive_train(stat_keys.copy(), functions, right_bin)

        result = [h_name, left_result, right_result]
        return result

    def _completely_classifeid(self, games):
        if not games:
            return True
        outcome = games[0][2]
        for game in games:
            if game[2] != outcome:
                return False
        return True

    def _calc_outcome(self, games):
        loss = 0
        win = 0
        for game in games:
            if game[2] == "Loss":
                loss += 1
            else:
                win += 1
        if loss > win:
            return "Loss"
        return "Win"

    def _calc_entropy(self, stat_keys, functions, all_games):
        h = []
        m = len(all_games)
        for key in stat_keys:
            mx = 0
            m1 = 0
            m1w = 0
            m2 = 0
            m2w = 0
            for game in all_games:
                year, team1, win1, team2, home = game
                heuristic = functions[key]
                result = heuristic(team1, year, team2, year, self.ps, self.ps, home)
                if win1 == "Loss":
                    mx += 1

                if result == "Win":
                    m1 += 1
                    if win1 == "Loss":
                        m1w += 1
                else:
                    m2 += 1
                    if win1 == "Loss":
                        m2w += 1

            e1 = entropy(mx, m)
            e2 = entropy(m - mx, m)
            e1i1 = entropy(m1w, m1)
            e1i2 = entropy(m1 - m1w, m1)
            e2i1 = entropy(m2w, m2)
            e2i2 = entropy(m2 - m2w, m2)
            i1 = (m1/m)*(e1i1 + e1i2)
            i2 = (m2/m)*(e2i1 + e2i2)
            e = (e1 + e2) - (i1 + i2)
            h.append([e, key])

        h.sort(reverse=True)
        if h[0][0] == 0:
            return None
        best_key = h[0][1]
        return best_key

    def build_up_collection(self, years, ps=False):
        all_games = []

        for year in years:
            for team1 in team_name_mappings:
                season = get_team_schedule(team1, year, post_season=ps)
                for game in season["outcome"]:
                    team2 = team_name_mappings2[game["opponent"]]
                    team1_win = game["win"]
                    home = game["home"]
                    result = [year, team1, team1_win, team2, home]
                    all_games.append(result)

        return all_games


class DecisionTreeCombinedAI:

    def __init__(self, years, post_season=False, pretrain=True):
        if pretrain:
            decisions = self.get_decisions()
        else:
            trainer = TreeTrainerCombinedAI()
            decisions = trainer.train(years, post_season)
        self.functions = get_heuristic_mapping()
        self.decisions = decisions

    def heuristic(self, team1, year1, team2, year2, ps1=False, ps2=False, home1=True):
        result = self.iter_tree(self.decisions, team1, year1, team2, year2, ps1, ps2, home1)
        return result

    def iter_tree(self, dtree, team1, year1, team2, year2, ps1, ps2, home1):
        if type(dtree) == str:
            return dtree

        h_name = dtree[0]
        heuristic = self.functions[h_name]
        result = heuristic(team1, year1, team2, year2, ps1, ps2, home1)

        if result != "Win":
            return self.iter_tree(dtree[1], team1, year1, team2, year2, ps1, ps2, home1)
        else:
            return self.iter_tree(dtree[2], team1, year1, team2, year2, ps1, ps2, home1)

    def get_decisions(self):
        return ['h_dtree', ['h_linear', ['h_basic', ['h_kth', 'Loss', 'Loss'], ['h_kth', 'Loss', 'Loss']],
                            ['h_basic', ['h_kth', 'Loss', 'Loss'], ['h_kth', 'Loss', 'Loss']]],
                ['h_linear', ['h_basic', ['h_kth', 'Win', 'Win'], ['h_kth', 'Win', 'Win']],
                 ['h_basic', ['h_kth', 'Win', 'Win'], ['h_kth', 'Win', 'Win']]]]


def heuristic_perceptron(team1, year1, team2, year2, ps1=False, ps2=False, home1=False):
    team1 = team_name_mappings[team1]["name1"]
    team2 = team_name_mappings[team2]["name1"]
    pts = float(get_offensive_stat(team1, year1, "PTS", ps1)) - float(get_offensive_stat(team2, year2, "PTS", ps2))
    yds = float(get_offensive_stat(team1, year1, "YDS", ps1)) - float(get_offensive_stat(team2, year2, "YDS", ps2))

    b = 69
    w = [-2835.3, 448.7]

    result = (w[0] * pts) + (w[1] * yds) + b
    if result < 0:
        return "Loss"
    else:
        return "Win"


dtreeCombined = DecisionTreeCombinedAI(training_years, pretrain=True)







