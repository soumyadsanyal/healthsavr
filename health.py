from bs4 import BeautifulSoup
import requests
import csv
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import statsmodels.api as sm
from imp import reload
import datetime
import sklearn
import pickle
from sklearn.externals import joblib
from scipy.stats import gaussian_kde
import pymysql as db
matplotlib.use('Agg')


def theurl(name):
    answer = {
        "2012": "http://meps.ahrq.gov/mepsweb/data_stats/download_data_files_codebook.jsp?PUFId=H155",
        "2013": "http://meps.ahrq.gov/mepsweb/data_stats/download_data_files_codebook.jsp?PUFId=H163",
        "population": "http://meps.ahrq.gov/mepsweb/data_stats/download_data_files_codebook.jsp?PUFId=H157",
        "medical": "http://meps.ahrq.gov/mepsweb/data_stats/download_data_files_codebook.jsp?PUFId=H154",
        "risk": "http://meps.ahrq.gov/mepsweb/data_stats/download_data_files_codebook.jsp?PUFId=H140",
        "employment": "http://meps.ahrq.gov/mepsweb/data_stats/download_data_files_codebook.jsp?PUFId=H131",
        "jobs": "http://meps.ahrq.gov/mepsweb/data_stats/download_data_files_codebook.jsp?PUFId=H158",
     "person_round_plan": "http://meps.ahrq.gov/mepsweb/data_stats/download_data_files_codebook.jsp?PUFId=H153"}
    return answer[name]


def make_soup(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.content)
    return soup


def get_header(name, target):
    return get_data(make_soup(theurl(name)), target)


def get_variables(target):
    with open(target, "r") as f:
        vars = f.read()
    vars = vars.splitlines()
    result = [(term.upper().replace("'", "'")).replace("-", "-")
              for term in vars]
    return result


def get_data(soup, target):
    result = []
    for row in soup.find_all("font"):
        result.append(row.contents)
    final = [(term[0].replace(u'\xa0', u'')).replace(',', '')
             for term in result]
# get rid of junk
    final = prune_list(final, "::")
    final = prune_list(final, "MEPS HC-155")
    final = prune_list(final, "MEPS HC-163")
    final = prune_list(final, "MEPS HC-157")
    final = prune_list(final, "MEPS H154 CODEBOOK")
    final = prune_list(final, "MEPS H140 CODEBOOK")
    final = prune_list(final, "MEPS H131 CODEBOOK")
    final = prune_list(final, "MEPS HC-150")
    final = prune_list(final, "MEPS HC-158")
    final = prune_list(final, "MEPS H153 CODEBOOK")
    final = prune_list(final, "2012 FULL YEAR CONSOLIDATED DATA CODEBOOK")
    final = prune_list(final, "2013 FULL YEAR CONSOLIDATED DATA CODEBOOK")
    final = prune_list(
        final, "2013 FULL YEAR POPULATION CHARACTERISTICS CODEBOOK")
    final = prune_list(final, "2012 MEPS MEDICAL CONDITIONS FILE")
    final = prune_list(final, "2002-2009 RISK ADJUSTMENT SCORES FILE")
    final = prune_list(final, "EMPLOYMENT VARIABLE IMPUTATION FILE")
    final = prune_list(final, "2012 JOBS FILE CODEBOOK")
    final = prune_list(final, "2013 JOBS FILE CODEBOOK")
    final = prune_list(final, "2012 PERSON ROUND PLAN FILE")
    final = prune_list(final, "DATE:   August 25 2014")
    final = prune_list(final, "DATE:   August 25 2015")
    final = prune_list(final, "DATE: August 21 2014")
    final = prune_list(final, "DATE: August 4 2014")
    final = prune_list(final, "DATE:     March 6 2015")
    final = prune_list(final, "DATE: December 15 2014")
    final = prune_list(final, "DATE:    April 10 2013")
    final = prune_list(final, "DATE:   August 12 2014")
    final = prune_list(final, "DATE: February 13 2015")
    done = final
    with open(target, "w") as f:
        f.write("start,end,variable\n")
        for skip in range(0, len(final)-1, 3):
            f.write("%s," % final[skip])
            f.write("%s," % final[skip+1])
            f.write("%s\n" % ((final[skip+2]).lstrip()).rstrip())
    print("Done")
    temp = pd.read_csv(target)
    temp = temp.sort("start").copy()
    temp.index = list(range(len(temp)))
    return temp


def prune_list(thelist, theterm):
    while True:
        try:
            thelist.pop(thelist.index(theterm))
        except ValueError:
            break
    return thelist


def pull_ascii_data(source):
    with open(source, 'r') as f:
        result = f.read()
    return result.split('\n')


def make_frame(data, header):
    dictionary = {header["variable"].ix[place]: [row[header["start"].ix[
        place]:header["end"].ix[place]] for row in data] for place in header.index}
    return pd.DataFrame(dictionary)


def all_together_now(datafile, headerfile):
    a = make_soup(theurl())
    header = get_data(a, headerfile)
    header.sort("start")
    data = pull_ascii_data(datafile)
    data = prune_list(data, '')
    return (data, header)


def write_table(data, header, target, short="No"):
    if short != "No":
        data = data[:10]
    data = prune_list(data, '')
    header = header.sort("start").copy()
    header.index = list(range(len(header)))
    with open(target, "w") as f:
        for element in header["variable"]:
            f.write("%s, " % element)
        f.write("\n")
    for row in data:
        u = [
            row
            [(header["start"].iloc[place] - 1): (header["end"].iloc[place])]
            for place in header.index]
        with open(target, "a") as f:
            writer = csv.writer(f)
            writer.writerow(u)
        print("Done with row %s" % data.index(row))


def swap_columns(theframe, here, there):
    temp = theframe[here].copy()
    theframe[here] = theframe[there].copy()
    theframe[there] = temp.copy()
    return theframe


def clean_columns(theframe):
    temp = theframe.columns.map(lambda x: (((str(x).lstrip()).rstrip()))).copy()
    theframe.columns = temp.copy()
    return theframe


def construct_dataset():
    cons = pd.read_csv("./consolidated_frame.csv")
    print("Read data")
    cons = clean_columns(cons)
    print("cleaned columns")
    office = cons["TOTAL OFFICE-BASED EXP 12"]
    print("office")
    outpatient = cons["TOTAL OUTPATIENT PROVIDER EXP 12"]
    print("outpatient")
    weight = cons["FINAL PERSON WEIGHT 2012"]
    print("weight")
    er = cons["TOTAL ER FACILITY + DR EXP 12"]
    print("er")
    inpatient = cons["TOT HOSP IP FACILITY + DR EXP 12"]
    print("inpatient")
    age = cons["AGE AS OF 12/31/12 (EDITED/IMPUTED)"]
    print("age")
    sex = cons["SEX"]
    print("sex")
    blood_pressure = cons["HIGH BLOOD PRESSURE DIAG (>17)"]
    print("blood_pressure")
    income = cons["FAMILY'S TOTAL INCOME"]
    print("income")
    married = cons["MARITAL STATUS-12/31/12 (EDITED/IMPUTED)"]
    print("married")
    drug = cons["TOTAL RX-EXP 12"]
    print("drug")
    region = cons["CENSUS REGION AS OF 12/31/12"]
    print("region")
    weight = cons["FINAL PERSON WEIGHT 2012"]
    print("weight")
    design_variables = [
        x.name for x in [
            married,
            income,
            blood_pressure,
            sex,
            age,
            er,
            inpatient,
            outpatient,
            office,
            region,
            drug,
         weight]]
    print("design_variables")
    design_matrix = cons[design_variables]
    print("design_matrix")
    X = design_matrix[design_matrix["FAMILY\'S TOTAL INCOME"] > 0]
    print("constructed X")
    X = X[X["AGE AS OF 12/31/12 (EDITED/IMPUTED)"] >= 0]
    print("cleaning age")
    X = X[X["HIGH BLOOD PRESSURE DIAG (>17)"] >= -1]
    print("cleaning hyp")
    X = X[X["MARITAL STATUS-12/31/12 (EDITED/IMPUTED)"] < 6]
    print("cleaning marital")
    X = X[X["CENSUS REGION AS OF 12/31/12"] > 0]
    print(" region")
    X = X[X["FINAL PERSON WEIGHT 2012"] > 0]
    print("cleaning weight")
    X["northeast"] = (X["CENSUS REGION AS OF 12/31/12"] == 1).astype(int)
    print("northeast")
    X["midwest"] = (X["CENSUS REGION AS OF 12/31/12"] == 2).astype(int)
    print("midwest")
    X["south"] = (X["CENSUS REGION AS OF 12/31/12"] == 3).astype(int)
    print("south")
    X["hypertension"] = (X["HIGH BLOOD PRESSURE DIAG (>17)"] == 1).astype(int)
    print("hypertension")
    X["male"] = (X["SEX"] == 1).astype(int)
    print("male")
    X["no longer married"] = (
        X["MARITAL STATUS-12/31/12 (EDITED/IMPUTED)"].map(lambda x: x in [2, 3, 4])).astype(int)
    print("no longer married")
    X["married"] = (
        X["MARITAL STATUS-12/31/12 (EDITED/IMPUTED)"].map(lambda x: x == 1)).astype(int)
    print("married")
    X["input weight"] = 1/X["FINAL PERSON WEIGHT 2012"]
    input_weights = X["input weight"]
    print("input weight")
    X.index = list(range(len(X)))
    print("reindexed X")
    X = split_dataset(X)
    print("annotate train, test, validate")
    exog_variables = [
        "married",
        "no longer married",
        "FAMILY'S TOTAL INCOME",
        "hypertension",
        "male",
        "AGE AS OF 12/31/12 (EDITED/IMPUTED)",
        "northeast",
        "midwest",
     "south"]
    print("assigned exog variables")
    endog_labels = {
        "office": ["TOTAL OFFICE-BASED EXP 12"],
        "outpatient": ["TOTAL OUTPATIENT PROVIDER EXP 12"],
        "inpatient": ["TOT HOSP IP FACILITY + DR EXP 12"],
     "er": ["TOTAL ER FACILITY + DR EXP 12"]}
    print("assigned endog labels")
    final = {}
    final["data"] = X
    final["exog_variables"] = exog_variables
    final["endog_labels"] = endog_labels
    final["input weights"] = input_weights
    final = clean_columns(final)
    return final


def split_dataset(X, name):
    random = pd.DataFrame(np.random.uniform(0, 1, len(X)))
    trainmask = random < .6
    testmask = (random < .8) & (random >= .6)
    validatemask = random >= .8
    X["train"] = trainmask.astype(int).copy()
    X["test"] = testmask.astype(int).copy()
    X["validate"] = validatemask.astype(int).copy()
    X.to_csv("./%s.csv" % name, index=False)
    X.to_pickle("./%s.pkl" % name)
    return X


def make_dummies(data, categorical_regressors, name):
    for variable in categorical_regressors:
        temp = list(data[variable].unique())
        dummies = [level for level in temp if level >= 0]
        dummies.pop()
        for level in dummies:
            data["%s==%s" % (variable, level)] = (
                data[variable] == level).astype(int)
    data.to_pickle("./%s" % name)
    return data


def trim(data, dependent_name, regressors_names, weight_name):
    temp = data.copy()
    temp = temp[temp[dependent_name] >= 0]
    temp = temp[temp[weight_name] > 0]
    for variable in regressors_names:
        temp = temp[temp[variable] >= 0]
    return temp


def compute_oop(pars, person):
    return np.dot(pars, person)


def process(person=[1, 1, 0, 90000, 1, 1, 55, 0, 0, 1]):
    this = construct_dataset()
    data = this["data"]
    exog_variables = this["exog_variables"]
    endog_labels = this["endog_labels"]
    theweights = this["input weights"]
    res = modeling(data, endog_labels, exog_variables)
    pars = {key: (res[key]).params for key in res.keys()}
    return {key: compute_oop(pars[key], person) for key in pars.keys()}


def read_raw_data():
    return pd.read_csv("./consolidated_frame.csv")


def normalize(matrix):
    return ((matrix-np.mean(matrix))/np.std(matrix))


def testing(data, yname, xnames, w):
    candidates = get_variables("./candidates.txt")


def rmse(model, test_set, y_name, location_regressors):
    y = test_set[y_name]
    exog = get_variables(location_regressors)
    y_hat = model.predict(test_set[exog])
    delta = y-y_hat
    squared_error = np.dot(delta, delta)
    mean_squared_error = float(squared_error)/len(y)
    root_mean_square_error = np.sqrt(mean_squared_error)
    return (
        "rmse = %s" %
        root_mean_square_error,
        "stddev in data = %s" %
        np.std(
            test_set[y_name]))


def wls_modeling(data, y_name, candidates_location, w_name, thealpha):
    # temp=data.copy()
    # print("made temp copy")
    candidates = get_variables("%s" % candidates_location)
    print("got candidates for regressors")
# temp=trim(temp,y_name,candidates,w_name)
    print("trimmed dataset")
# model=sm.WLS(temp[y_name],sm.add_constant(temp[candidates]),1./temp[w_name])
    model = sm.WLS(
        data[y_name],
        sm.add_constant(
            data[candidates]),
         1./data[w_name])
    print("assigned model")
    res = model.fit_regularized(alpha=thealpha)
    print("fit model")
    res.save("./%swls_model%s.pkl" % (y_name, datetime.datetime.today()))
    print("saved model")
    return res


def rf_trim(data, dependent_name, regressors_names):
    temp = data.copy()
    temp = temp[temp[dependent_name] >= 0]
    for variable in regressors_names:
        temp = temp[temp[variable] >= 0]
    return temp


def rf_modeling(data, y_name, candidates_location, n_trees, w_name):
    from sklearn.ensemble import RandomForestRegressor
    temp = data.copy()
    print("made temp copy")
    candidates = get_variables("./%s" % candidates_location)
    print("got candidates for regressors")
# temp=rf_trim(temp,y_name,candidates)
# print("trimmed dataset")
    model = RandomForestRegressor(
        n_estimators=n_trees,
        min_samples_split=2,
     oob_score=True)
    print("assigned model")
    res = model.fit(
        temp[candidates],
        temp[y_name],
        sample_weight=np.asarray(
            temp[w_name]))
    print("fit model")
    print("saved model")
    importance = sorted([(res.feature_importances_[place],
                          candidates[place]) for place in range(
                             len(candidates))])
    importance.reverse()
    with open(candidates_location, "w") as f:
        for term in importance:
            f.write("%s\n" % term[1])
    print("made importance list")
    return (res)


def pickle_model(themodel, location):
    from sklearn.externals import joblib
    joblib.dump(themodel, location)
    print("saved model")


def load_model(kind):
    from sklearn.externals import joblib
    location = "rf_%s_bucketed.pkl" % kind
    return joblib.load(location)


def gbr_modeling(data, y_name, candidates_location, n_trees, w_name):
    from sklearn.ensemble import GradientBoostingRegressor
    temp = data.copy()
    print("made temp copy")
    candidates = get_variables("./%s" % candidates_location)
    print("got candidates for regressors")
# temp=rf_trim(temp,y_name,candidates)
# print("trimmed dataset")
    model = GradientBoostingRegressor(n_estimators=n_trees, min_samples_split=2)
    print("assigned model")
    res = model.fit(
        temp[candidates],
        temp[y_name],
        sample_weight=np.asarray(
            temp[w_name]))
    print("fit model")
# joblib.dump(res,"./%srf_model%s.pkl"%(y_name,datetime.datetime.today()))
# print("saved model")
    return res


def gb_modeling(data, y_name, candidates_location, n_trees, w_name):
    from sklearn.ensemble import GradientBoostingRegressor
    temp = data.copy()
    print("made temp copy")
    candidates = get_variables("./%s" % candidates_location)
    print("got candidates for regressors")
# temp=rf_trim(temp,y_name,candidates)
# print("trimmed dataset")
    model = GradientBoostingRegressor(n_estimators=n_trees, min_samples_split=1)
    print("assigned model")
    res = model.fit(
        temp[candidates],
        temp[y_name],
        sample_weight=np.asarray(
            temp[w_name]))
    print("fit model")
    joblib.dump(res, "./%sgb_model%s.pkl" % (y_name, datetime.datetime.today()))
    print("saved model")
    return res


def br_modeling(data, y_name, candidates_location):
    from sklearn.linear_model import BayesianRidge
    temp = data.copy()
    print("made temp copy")
    candidates = get_variables("./%s" % candidates_location)
    print("got candidates for regressors")
    temp = rf_trim(temp, y_name, candidates)
    print("trimmed dataset")
    model = BayesianRidge()
    print("assigned model")
    res = model.fit(temp[candidates], temp[y_name])
    print("fit model")
    joblib.dump(res, "./%sbr_model%s.pkl" % (y_name, datetime.datetime.today()))
    print("saved model")
    return res


def get_persons_and_insurance():
    # meps_modeled=pd.read_pickle("./meps_bucketed.pkl")
    meps_modeled = pd.read_pickle(
        "./all_with_estimates_and_buckets_20150929.pkl")
    print("got meps")
    insurance = pd.read_pickle("../data/insurance_current_.pkl")
    print("got insurance")
    exog = get_all_regressors()
    print("got exog")
    return (meps_modeled, exog, insurance)


def underscore_the_variables(location):
    with open(location, "r") as f:
        temp = f.read()
    temp = temp.splitlines()
    result = [term.replace(' ', '_') for term in temp]
    with open(location.replace(".txt", "_.txt"), "w") as f:
        for term in result:
            f.write("%s\n" % term)


def underscore_the_headers(data):
    temp = (data.columns).copy()
    result = [term.replace(' ', '_') for term in temp]
    data.columns = result.copy()
    return data


def underscore_the_headers_in_pickled_files(location):
    from sklearn.externals import joblib
    data = pd.read_pickle(location)
    data = underscore_the_headers(data)
    data.to_pickle("%s" % (location.replace(".pkl", "_.pkl")))
    print("done")


def prep_for_modeling():
    all = pd.read_pickle("./this_is_the_set_i_built_the_models_on_.pkl")
    print("got data")
# cats=get_variables("./categorical_regressors__.txt")
# print("got categorical variables")
# all=make_dummies(all,cats,"2013_with_dummies_.pkl")
# print("made dummies")
    with open("./exog_rf_.txt", "r") as f:
        temp = f.read()
        exog = temp.splitlines()
    print("got exog")
    all = bucketize(all, exog)
    print("bucketized all regressors in all")
    with open("./exog_rf__.txt", "r") as f:
        temp = f.read()
        exog = temp.splitlines()
    train = all[all["train"] == 1]
    print("made training set")
    test = all[all["test"] == 1]
    print("made test set")
    validate = all[all["validate"] == 1]
    print("made set")
    endog = get_variables("./expenses13_.txt")
    print("got endog")
    office = endog[0]
    print("assigned office")
    outpatient = endog[1]
    print("assigned outpatient")
    er = endog[2]
    print("assigned er")
    inpatient = endog[3]
    print("assigned inpatient")
    w = get_variables("weights13_.txt")
    print("assigned w")
    w = w[0]
    print("popped w")
    insurance = pd.read_pickle("../data/insurance_current_.pkl")
    print("got insurance plans")
# all_with_estimated_charges=pd.read_pickle("./meps_with_buckets.pkl")
# all_with_estimated_charges=pd.read_pickle("./all_with_estimated_charges_underscored.pkl")
# print("got all data with estimated costs and buckets")
# return (all, train, test, validate, endog, office, outpatient,
# inpatient, er, w, exog, insurance, all_with_estimated_charges)
    return (
        all,
        train,
        test,
        validate,
        endog,
        office,
        outpatient,
        inpatient,
        er,
        w,
        exog,
     insurance)


def break_insurance(insurance):
    for key in list(insurance.columns[5:10]):
        insurance[
            "breakout of %s" %
            key] = insurance[key].map(
            lambda x: breakout_insurance(x))
    pcpbreak = insurance.columns[-4]
    specialistbreak = insurance.columns[-3]
    inpatientdocbreak = insurance.columns[-2]
    emergencybreak = insurance.columns[-1]
    breaks = [pcpbreak, specialistbreak, inpatientdocbreak, emergencybreak]
    insurance["pcp copay before deductible"] = insurance[
        pcpbreak].map(lambda x: x[0][0][1])
    insurance["pcp coinsurance before deductible"] = insurance[
        pcpbreak].map(lambda x: x[0][1][1])
    insurance["pcp copay after deductible"] = insurance[
        pcpbreak].map(lambda x: x[1][0][1])
    insurance["pcp coinsurance after deductible"] = insurance[
        pcpbreak].map(lambda x: x[1][1][1])
    insurance["specialist copay before deductible"] = insurance[
        specialistbreak].map(lambda x: x[0][0][1])
    insurance["specialist copay before deductible"] = insurance[
        specialistbreak].map(lambda x: x[0][0][1])
    insurance["specialist coinsurance before deductible"] = insurance[
        specialistbreak].map(lambda x: x[0][1][1])
    insurance["specialist copay after deductible"] = insurance[
        specialistbreak].map(lambda x: x[1][0][1])
    insurance["specialist coinsurance after deductible"] = insurance[
        specialistbreak].map(lambda x: x[1][1][1])
    insurance["inpatientdoc copay before deductible"] = insurance[
        inpatientdocbreak].map(lambda x: x[0][0][1])
    insurance["inpatientdoc coinsurance before deductible"] = insurance[
        inpatientdocbreak].map(lambda x: x[0][1][1])
    insurance["inpatientdoc copay after deductible"] = insurance[
        inpatientdocbreak].map(lambda x: x[1][0][1])
    insurance["inpatientdoc coinsurance after deductible"] = insurance[
        inpatientdocbreak].map(lambda x: x[1][1][1])
    insurance["emergency copay before deductible"] = insurance[
        emergencybreak].map(lambda x: x[0][0][1])
    insurance["emergency coinsurance before deductible"] = insurance[
        emergencybreak].map(lambda x: x[0][1][1])
    insurance["emergency copay after deductible"] = insurance[
        emergencybreak].map(lambda x: x[1][0][1])
    insurance["emergency coinsurance after deductible"] = insurance[
        emergencybreak].map(lambda x: x[1][1][1])
    return insurance


def cv_modeling(estimator, data, y_name, candidates_location, n_trees, w_name):
    from sklearn.ensemble import estimator
    from sklearn.grid_search import GridSearchCV
    temp = data.copy()
    print("made temp copy")
    candidates = get_variables("./%s" % candidates_location)
    print("got candidates for regressors")
# temp=rf_trim(temp,y_name,candidates)
# print("trimmed dataset")
    model = estimator()
    print("assigned model")
    parameters = {}
    clf = GridSearchCV(model, parameters)
    res = clf.fit(temp[candidates], temp[y_name])
# print("fit model")
# joblib.dump(res,"./%sgb_model%s.pkl"%(y_name,datetime.datetime.today()))
# print("saved model")
    return res


def find_kernel(data):
    from sklearn.neighbors import KernelDensity
    freq = plt.hist(np.asarray(data), bins=len(data), normed=True)
# np.random.seed(1)
# X = freq[0][:,np.newaxis]
# X_plot = np.linspace(-5, 10, len(X))[:, np.newaxis]
    fig, ax = plt.subplots()
# kde = KernelDensity(kernel="gaussian", bandwidth=bw).fit(X)
# log_dens = kde.score_samples(X_plot)
# ax.plot(X_plot[:, 0], np.exp(log_dens), '-')
    max_y = max(freq[0])
    temp = [int(term > 0) for term in freq[0]]
    temp_rev = temp[::-1]
    max_x = temp_rev.index(1)
    ax.set_xlim((0, max_x))
    ax.set_ylim((0, max_y))
    plt.show()
    return (freq, max_x, max_y)
# return (kde, X, X_plot, log_dens)


def breakout_insurance(thestring):
    return before_after(thestring)


def before_after(thestring):

    if ("before" in thestring) and ("after" in thestring):

        (this, that) = thestring.split("and")
        return list(map(copay_coinsurance, (["before", this], ["after", that])))

    if ("before" in thestring) and not ("after" in thestring):

        return list(map(copay_coinsurance, (["before", thestring], [
                    "after", "$0 copay and 0% coinsurance"])))

    if not ("before" in thestring) and ("after" in thestring):

        return list(map(copay_coinsurance, ([
                    "before", "$0 copay and 100% coinsurance"], ["after", thestring])))

    if not ("before" in thestring) and not ("after" in thestring):

        return list(
            map(copay_coinsurance, (["before", thestring], ["after", thestring])))

    else:

        raise ValueError("this is probably not a valid policy")


def copay_coinsurance(pair):
    (thisstring, thatstring) = pair
    if thisstring == "before":

        if ("copay" in thatstring) and ("coinsurance" in thatstring):
            return list(map(extract_cost_sharing, thatstring.split("and")))

        if not ("copay" in thatstring) and ("coinsurance" in thatstring):
            return list(map(extract_cost_sharing, ("$0 copay", thatstring)))

        if ("copay" in thatstring) and not ("coinsurance" in thatstring):
            return list(map(extract_cost_sharing,
                            (thatstring, "100% coinsurance")))

        if not ("copay" in thatstring) and not ("coinsurance" in thatstring):
            return copay_coinsurance(["before", annotate(thatstring)])

        else:
            raise ValueError("neither copay nor coinsurance before deductible")

    if thisstring == "after":

        if ("copay" in thatstring) and ("coinsurance" in thatstring):
            return list(map(extract_cost_sharing, thatstring.split("and")))

        if not ("copay" in thatstring) and ("coinsurance" in thatstring):
            return list(map(extract_cost_sharing, ("$0 copay", thatstring)))

        if ("copay" in thatstring) and not ("coinsurance" in thatstring):
            return list(map(extract_cost_sharing,
                            (thatstring, "0% coinsurance")))

        if not ("copay" in thatstring) and not ("coinsurance" in thatstring):
            return copay_coinsurance(["after", annotate(thatstring)])

        else:
            raise ValueError("neither copay nor coinsurance after deductible")

    else:
        raise ValueError(
            "this is neither before deductible nor after deductible")


def annotate(thestring):
    if ("$" in thestring) and ("%" in thestring):
        (first, second) = thestring.split("and")
        return ''.join([annotate(first), "and", annotate(second)])
    if ("$" in thestring):
        return ("%s copay " % thestring)
    if "%" in thestring:
        return ("%s coinsurance " % thestring)
    if thestring.lower() == "no charge after deductible" or thestring.lower() == "no charge":
        return "$0 copay and 0% coinsurance after deductible"
    else:
        raise ValueError("failed to annotate %s" % thestring)


def extract_cost_sharing(thestring):
    return extract_parameter(thestring)


def extract_parameter(thestring):
    thestring = thestring.lower()
    if "copay" in thestring:
        return [("copay", int(term)) for term in (
            (thestring.replace("$", "")).replace("%", "")).split() if term.isdigit()].pop()
    if "coinsurance" in thestring:
        return [("coinsurance",
                 int(term)/100) for term in ((thestring.replace("$",
                                                                "")).replace("%",
                                                                             "")).split() if term.isdigit()].pop()
    if thestring == "no charge after deductible" or thestring == "no charge":
        return [("whatever", 0)].pop()
    if "$" in thestring:
        return [("copay", int(term)) for term in (
            (thestring.replace("$", "")).replace("%", "")).split() if term.isdigit()].pop()
    if "%" in thestring:
        return [("coinsurance",
                 int(term)/100) for term in ((thestring.replace("$",
                                                                "")).replace("%",
                                                                             "")).split() if term.isdigit()].pop()
    else:
        raise ValueError(
            "%s is probably not a valid cost sharing policy" %
            thestring)


def get_all_regressors():
    result = []
    for kind in ["office", "outpatient", "inpatient", "er"]:
        exog = get_variables("./exog_building_rf_%s__.txt" % kind)
        for term in exog:
            result.append(term)
    return list(set(result))


def get_levels(data, exog):
    g = {}
    for term in exog:
        temp = sorted(data[term].unique())
        g[term] = temp
    return g


def bucket(regressor, value):
    if "ADULT" in regressor and "BODY" in regressor and "MASS" in regressor:
        return int((value > 18.5)) + int((value > 24.9)) + int((value > 29.9))
    if "AGE_OF_DIAGNOSIS" in regressor:
        return int((value > 13)) + int((value > 19)) + int((value > 25)) + int((value
                                                                                > 30)) + int((value > 40)) + int((value > 50)) + int((value > 60))
    if "DIAG" in regressor and not ("AGE" in regressor):
        return int((value > 0)) + int((value == 1))
    if "AGE_AS_OF_" in regressor:
        return int((value > 12)) + int((value > 18)) + int((value > 25)) + int((value
                                                                                > 30)) + int((value > 40)) + int((value > 50)) + int((value > 60))
    if "EDUCATION" in regressor:
        return int((value > 0)) + int((value > 1)) + int((value > 2)
                                                         ) + int((value > 13)) + int((value > 14)) + int((value > 15))
    if "PREGNANT" in regressor:
        return int((value > 0)) + int((value == 1))
    if "RACE" in regressor:
        return int((value > 0)) + int((value > 1)) + int((value > 2)
                                                         ) + int((value > 3)) + int((value > 4))
    if "PERSON" in regressor and "INCOME" in regressor:
        return int((value > 10000)) + int((value > 35000)) + int((value > 50000)) + int((value > 70000)
                                                                                        ) + int((value > 90000)) + int((value > 120000)) + int((value > 150000)) + int((value > 200000))
    if "FAMILY" in regressor and "INCOME" in regressor:
        return int((value > 18000)) + int((value > 60000)) + int((value > 90000)) + int(
            (value > 120000)) + int((value > 150000)) + int((value > 225000)) + int((value > 400000))
    if "CENSUS" in regressor:
        return int((value > 0)) + int((value > 1)
                                      ) + int((value > 2)) + int((value > 3))
    if "#" in regressor:
        return int((value > 4)) + int((value > 10)) + int((value > 15)) + int((value >
                                                                               20)) + int((value > 30)) + int((value > 40)) + int((value > 50))
    else:
        return value


def bucketize(data, exog):
    for term in exog:
        data["%s_BUCKET" % term] = data[term].map(lambda x: bucket(term, x))
    return data


def bucketize_exog_files(location):
    with open(location, "r") as f:
        temp = f.read()
        temp = temp.splitlines()
    if "_.txt" in location:
        with open(("%s" % location).replace("_.txt", "__.txt"), "w") as f:
            for term in temp:
                f.write("%s_BUCKET\n" % term)
    else:
        raise ValueError("location does not have _.txt in the name")


def add_estimated_charges_columns(data):
    for kind in ["office", "outpatient", "inpatient", "er"]:
        data["estimate_%s_charges" % kind] = estimate_billed_charges(data, kind)
    return data


def estimate_billed_charges(data, kind):
    exog = get_variables("./exog_building_rf_%s__.txt" % kind)
    temp = data[exog]
    print("made copy of data")
    model = load_model(kind)
    print("loaded model")
# model_outpatient=load_model("outpatient")
# print("loaded model 2")
# model_inpatient=load_model("inpatient")
# print("loaded model 3")
# model_er=load_model("er")
# print("loaded model 4")
# exog_office=get_variables("./exog_building_rf_office.txt")
# print("loaded exog 1")
# exog_outpatient=get_variables("./exog_building_rf_outpatient.txt")
# print("loaded exog 2")
# exog_inpatient=get_variables("./exog_building_rf_inpatient.txt")
# print("loaded exog 3")
# exog_er=get_variables("./exog_building_rf_er.txt")
# print("loaded exog 4")
# charges=[model_office.predict(temp[exog_office]), model_outpatient.predict(temp[exog_outpatient]), model_inpatient.predict(temp[exog_inpatient]), model_er.predict(temp[exog_er])]
# return (charges)
    charge = model.predict(temp)
    return charge


def get_insurance_breakout_from_plan_id(planid):
    con = db.connect("localhost", "root", "bebelus", "insurancedb")
    with con:
        cursor = con.cursor()
        cursor.execute(
            "select * from insurancedb where Plan_ID_LEFTPARstandard_componentRIGHTPAR = \"%s\" " %
            (planid))
        temp = cursor.fetchall()
        theplan = temp[0]
        deductible = theplan[2]
        max_oop = theplan[4]
        copay_after = theplan[17]
        coinsurance_after = theplan[18]
        copay_before = theplan[19]
        coinsurance_before = theplan[20]
    return (
        deductible,
        max_oop,
        copay_before,
        coinsurance_before,
        copay_after,
     coinsurance_after)


def estimate_oop_from_database(thedictionary, planone, plantwo):
    # thestring=''.join('{}{}'.format(key,val) for (key,val) in thedictionary.items())
    # print(thestring)
    # # input()
    search_string = "select estimate_office_charges, office_visits, FINAL_PERSON_WEIGHT_2013 from mepsdb where "
    d = {}
    for (key, value) in thedictionary.items():
        if value is not None and value != '':
            d[key] = value
    count = 0
    for (key, value) in d.items():
        if count == 0:
            search_string = search_string + "%s = %s" % (clean(key)[:64], value)
            count = count+1
        else:
            search_string = search_string + \
                " and %s = %s" % (clean(key)[:64], value)
# print(search_string)
# # input()
    con = db.connect("localhost", "root", "bebelus", "mepsdb")
    with con:
        cursor = con.cursor()
        cursor.execute(search_string)
        temp = cursor.fetchall()
        charges = []
        visits = []
        weights = []
        for row in temp:
            charges.append(row[0])
            visits.append(row[1])
            weights.append(row[2])
    temp = pd.DataFrame({"estimate_office_charges": charges,
                         "office_visits": visits,
                         "FINAL_PERSON_WEIGHT_2013": weights})
    plans = [planone, plantwo]
    oop_dict = {}
    html_dict = {}
    print("estimated visits")
    for plan in plans:
        kinds = ["office"]
# kinds=["office","outpatient","inpatient","er"]
        for kind in kinds:
            #   breakout=plan[plan.index[16+kinds.index(kind)]]
            #   print(breakout)
            #   deductible=plan[plan.index[1]]
            #   print(deductible)
            #   max_oop=plan[plan.index[3]]
            #   print(max_oop)
            (deductible, max_oop, copay_before, coinsurance_before, copay_after,
             coinsurance_after) = get_insurance_breakout_from_plan_id(plan)
            temp[
                "oop_%s" %
                kind] = temp.apply(
                lambda x: compute_oop(
                    x[0],
                    x[1],
                    deductible,
                    max_oop,
                    copay_before,
                    coinsurance_before,
                    copay_after,
                    coinsurance_after),
             axis=1)
            f = {}
            f["oop_%s" % kind] = plt.hist(np.asarray(temp["oop_%s" % kind]), bins=min(
                100, len(temp)), weights=np.asarray(temp["FINAL_PERSON_WEIGHT_2013"]), normed=True)
            plt.title(
                "Estimated out of pocket expenses for %s-based services under plan %s" %
                (kind, plan))
            plt.xlabel(
                "Estimated out of pocket expenses on %s-based services" %
                kind)
            plt.ylabel("Probability")
            plt.xlim(
                min(np.asarray(temp["oop_%s" % kind])),
                max(np.asarray(temp["oop_%s" % kind])))
            plt.ylim(min(f["oop_%s" % kind][0]), max(f["oop_%s" % kind][0]))
            plan_base = "static/images/histogram_of_oop_%s_under_%s.png" % (
                                                                            kind,
                                                                            plan)
            oop_dict[plan] = "app/" + plan_base
            html_dict[plan] = "../" + plan_base
            plt.savefig(oop_dict[plan])
            plt.clf()
            plt.close()
    f["%s_billed" % kind] = plt.hist(np.asarray(
            temp["estimate_%s_charges" % kind]),
        bins=min(500, len(temp)),
        weights=np.asarray(temp["FINAL_PERSON_WEIGHT_2013"]),
        normed=True)
    plt.title("Estimated billed charges for %s-based services" % (kind))
    plt.xlabel("Estimated billed charges for %s-based services" % kind)
    plt.ylabel("Probability")
    plt.xlim(min(np.asarray(temp["estimate_%s_charges" % kind])), max(
        np.asarray(temp["estimate_%s_charges" % kind])))
    plt.ylim(min(f["%s_billed" % kind][0]), max(f["%s_billed" % kind][0]))
    billed_base = "static/images/histogram_of_billed_charges_%s.png" % (kind)
    billed_html = "../"+billed_base
    billed = "app/" + billed_base
    plt.savefig(billed)
    plt.clf()
    plt.close()
    #  return f
    # with open("./app/templates/soumya.json","w") as f:
    #  f.write("[")
    #  for term in (np.asarray(temp["estimate_office_charges"]))[:-1]:
    #   f.write("%d,"%term)
    #  f.write("%d]\n"%(np.asarray(temp["estimate_office_charges"]))[-1])
    return (html_dict, billed_html)


def compute_oop(
    total_charged,
    num_visits,
    deductible,
    max_oop,
    copay_before,
    coinsurance_before,
    copay_after,
     coinsurance_after):
    if num_visits == 0:
        return 0
    if num_visits > 0:
        cpv = total_charged/num_visits
    if num_visits < 0:
        raise ValueError("negative number of visits")
# copay_before=breakout[0][0][1]
# coinsurance_before=breakout[0][1][1]
# copay_after=breakout[1][0][1]
# coinsurance_after=breakout[1][1][1]
    paid_before = copay_before + coinsurance_before*(max(0, cpv-copay_before))
    oop_before = coinsurance_before*(max(0, cpv-copay_before))
    length_before = min(num_visits, deductible/(1+oop_before))
    total_paid_before_deductible = paid_before*length_before
    charged_before = length_before*cpv
    charged_after = total_charged-charged_before
    visits_after = num_visits-length_before
    paid_after = copay_after + coinsurance_after*max(0, cpv-copay_after)
    oop_after = coinsurance_after*max(0, cpv-copay_after)
    length_after = min(visits_after, (max_oop-deductible)/(1+oop_after))
    total_paid_after_deductible = paid_after*length_after
# total_paid_after_deductible=(copay_after + coinsurance_after*max(0,cpv-copay_after))*((min(max(total_charged-deductible,0),max_oop-deductible))/(1+ coinsurance_after*max(0,cpv-copay_after)))
    print("total_charged = %f \n deductible=%f, \n maxoop=%f, \n visits=%f, \n cpv=%f, \n copay_before=%f, \n coinsurance_before=%f, \n copay_after=%f, \n coinsurance_after=%f, \n paid_before=%f, \n oop_before=%f, \n length_before=%f, \n total_paid_before_deductible=%f, \n charged_before=%f, \n charged_after=%f, \n visits_after=%f, \n paid_after=%f, \n oop_after=%f, \n length_after=%f, \n total_paid_after_deductible=%f" % (
        total_charged, deductible, max_oop, num_visits, cpv, copay_before, coinsurance_before, copay_after, coinsurance_after, paid_before, oop_before, length_before, total_paid_before_deductible, charged_before, charged_after, visits_after, paid_after, oop_after, length_after, total_paid_after_deductible))
# return (int(total_paid_before_deductible) ,
# int(total_paid_after_deductible),
# min(max_oop,int(total_paid_before_deductible +
# total_paid_after_deductible)))
    return min(
        max_oop, int(
            total_paid_before_deductible + total_paid_after_deductible))


# def estimate_oop(data,thedictionary,planone, plantwo):
# data=estimate_visits(data)
# plans=[planone, plantwo]
# oop_dict={}
# print("estimated visits")
# temp=cross_section(data,thedictionary)
# for plan in plans:
#  kinds=["office"]
# kinds=["office","outpatient","inpatient","er"]
#  for kind in kinds:
#   breakout=plan[plan.index[16+kinds.index(kind)]]
#   print(breakout)
#   deductible=plan[plan.index[1]]
#   print(deductible)
#   max_oop=plan[plan.index[3]]
#   print(max_oop)
#   temp["oop_%s"%kind]=temp[["estimate_%s_charges"%kind,"%s_visits"%kind]].apply(lambda x: compute_oop(x[0],x[1],breakout,deductible,max_oop ),axis=1)
#   f={}
#   f["oop_%s"%kind]=plt.hist(np.asarray(temp["oop_%s"%kind]),bins=min(100,len(temp)),weights=np.asarray(temp["FINAL_PERSON_WEIGHT_2013"]),normed=True)
#   plt.title("Estimated out of pocket expenses for %s-based services under plan %s"%(kind,plan[plan.index[0]]))
#   plt.xlabel("Estimated out of pocket expenses on %s-based services"%kind)
#   plt.ylabel("Probability")
#   plt.xlim(min(np.asarray(temp["oop_%s"%kind])),max(np.asarray(temp["oop_%s"%kind])))
#   plt.ylim(min(f["oop_%s"%kind][0]),max(f["oop_%s"%kind][0]))
#   oop_dict[plan["Plan_ID_(standard_component)"]]="./static/images/histogram_of_oop_%s_under_%s.png"%(kind,plan["Plan_ID_(standard_component)"])
#   plt.savefig(oop_dict[plan["Plan_ID_(standard_component)"]])
#   plt.clf()
#   plt.close()
# f["%s_billed"%kind]=plt.hist(np.asarray(temp["estimate_%s_charges"%kind]),bins=min(500,len(temp)),weights=np.asarray(temp["FINAL_PERSON_WEIGHT_2013"]),normed=True)
# plt.title("Estimated billed charges for %s-based services"%(kind))
# plt.xlabel("Estimated billed charges for %s-based services"%kind)
# plt.ylabel("Probability")
# plt.xlim(min(np.asarray(temp["estimate_%s_charges"%kind])),max(np.asarray(temp["estimate_%s_charges"%kind])))
# plt.ylim(min(f["%s_billed"%kind][0]),max(f["%s_billed"%kind][0]))
# billed="./static/images/histogram_of_billed_charges_%s.png"%(kind)
# plt.savefig(billed)
# plt.clf()
# plt.close()
# #  return f
# # with open("./app/templates/soumya.json","w") as f:
# #  f.write("[")
# #  for term in (np.asarray(temp["estimate_office_charges"]))[:-1]:
# #   f.write("%d,"%term)
# #  f.write("%d]\n"%(np.asarray(temp["estimate_office_charges"]))[-1])
# return (oop_dict,billed)

# def compute_oop(total_charged, num_visits, breakout, deductible, max_oop):
# if num_visits==0:
#  return 0
# if num_visits>0:
#  cpv=total_charged/num_visits
# if num_visits<0:
#  raise ValueError("negative number of visits")
# copay_before=breakout[0][0][1]
# coinsurance_before=breakout[0][1][1]
# copay_after=breakout[1][0][1]
# coinsurance_after=breakout[1][1][1]
# total_paid_before_deductible=min(deductible,total_charged*coinsurance_before) + (min(deductible,total_charged)/cpv)*copay_before
# total_paid_after_deductible=(max(0,min(max_oop-deductible,total_charged-deductible))/cpv)*copay_after + min(max_oop-deductible,(total_charged-deductible)*coinsurance_after)
# total_paid_after_deductible=(copay_after + coinsurance_after*max(0,cpv-copay_after))*((min(max(total_charged-deductible,0),max_oop-deductible))/(1+ coinsurance_after*max(0,cpv-copay_after)))
# print("total_charged = %f \n deductible=%f, \n maxoop=%f, \n visits=%f, \n cpv=%f, \n copay_before=%f, \n coinsurance_before=%f, \n copay_after=%f, \n coinsurance_after=%f, \n paid_before=%f, \n oop_before=%f, \n length_before=%f, \n total_paid_before_deductible=%f, \n charged_before=%f, \n charged_after=%f, \n visits_after=%f, \n paid_after=%f, \n oop_after=%f, \n length_after=%f, \n total_paid_after_deductible=%f"%(total_charged, deductible, max_oop, num_visits, cpv, copay_before, coinsurance_before, copay_after, coinsurance_after, paid_before, oop_before, length_before, total_paid_before_deductible, charged_before, charged_after, visits_after, paid_after, oop_after, length_after, total_paid_after_deductible))
# return (int(total_paid_before_deductible) , int(total_paid_after_deductible), min(max_oop,int(total_paid_before_deductible + total_paid_after_deductible)))
# return min(max_oop,int(total_paid_before_deductible +
# total_paid_after_deductible))


def clean(thestring):
    return (
        (
            ((((((((((((thestring.replace(
                "\'",
                "APOS")).replace(
                "#",
                "NUM")).replace(
                    "+",
                    "PLUS")).replace(
                        "/",
                        "FORSLASH")).replace(
                            ">",
                            "GTHAN")).replace(
                                "<",
                                "LTHAN")).replace(
                                    "(",
                                    "LEFTPAR")).replace(
                                        ")",
                                        "RIGHTPAR")).replace(
                                            "-",
                                            "DASH")).replace(
                                                ":",
                                                "COLON")).replace(
                                                    ".",
                                                    "FULLSTOP")).replace(
                                                        "&",
                                                        "AMPERSAND")).replace(
                                                            "=",
                                                            "EQUALSIGN")).replace(
                                                                "%",
                                                                 "PERCENTSIGN"))


def dirty(thestring):
    return ((((((thestring.replace("APOS", "\'")).replace("NUM", "#")).replace(
        "PLUS", "+")).replace("FORSLASH", "/")).replace("GTHAN", ">")).replace("LTHAN", "<"))


def get_user_values(exog_list):
    result = {}
    for term in exog_list:
        result[term] = input("Enter value for %s: " % (term))
    return result


def cross_section(thedata, thedictionary):
    temp = thedata
    for key in thedictionary.keys():
        if thedictionary[key] != '':
            temp = temp[temp["%s" % key] == (int(thedictionary[key]))]
            print(len(temp))
    if len(temp) > 0:
        return temp
    else:
        raise ValueError("There are no data points matching your criteria")


def estimate_visits(data):
    data["office_visits"] = data["#_OFFICE-BASED_PROVIDER_VISITS_13"]
    data["outpatient_visits"] = data["#_OUTPATIENT_DEPT_PROVIDER_VISITS_13"]
    data["inpatient_visits"] = data["#_NIGHTS_IN_HOSP_FOR_DISCHARGES_2013"]
    data["er_visits"] = data["#_EMERGENCY_ROOM_VISITS_13"]
    return data


def unpack_visits(coded):
    if coded == 0:
        return 2
    if coded == 1:
        return 8
    if coded == 2:
        return 13
    if coded == 3:
        return 18
    if coded == 4:
        return 25
    if coded == 5:
        return 35
    else:
        pass


# def join_copay_coinsurance(thelist):
# result=[]
# for term in thelist[:-1]:
#  result.append("%s and "%annotate(term))
#  if "$" in term:
#   result.append("%s copay "%term)
#  if "%" in term:
#   result.append("%s coinsurance "%term)
# result.append(annotate(thelist[-1]))
# final=''.join(tuple(result))
# return final


# def split_on_deductible(thestring):
# thestring=thestring.lower()
# if "before" in thestring and "after" in thestring:
#  return list(map(extract_copay_coinsurance,thestring.split("and")))
# if "after" in thestring:
#  if "copay" in thestring and "coinsurance" in the string:
#   return list(map(extract_copay_coinsurance,("100% coinsurance",thestrin
#  if "copay" in thestring and not ("coinsurance" in the string):
#   return list(map(extract_copay_coinsurance,(thestring,"100% coinsurance")))
#  if not ("copay" in thestring) and ("coinsurance" in the string):
#   return list(map(extract_copay_coinsurance,("$0 copay",thestring)))

#  return list(map(extract_copay_coinsurance,("100% coinsurance",thestring)))
# else:
#  if "copay" in thestring and "coinsurance" in the string:
#   return list(map(extract_copay_coinsurance,(thestringsplit.("and"))))
#  if "copay" in thestring and not ("coinsurance" in the string):
#   return list(map(extract_copay_coinsurance,(thestring,"100% coinsurance")))
#  if not ("copay" in thestring) and ("coinsurance" in the string):
#   return list(map(extract_copay_coinsurance,("$0 copay",thestring)))


# def extract_copay_coinsurance(thestring):
# thestring=thestring.lower()
# if "copay" in thestring and "coinsurance" in thestring:
#  return list(map(extract_parameter,thestring.split("and")))
# if "copay" in thestring:
#  return list(map(extract_parameter,(thestring,"0% coinsurance")))
# if "coinsurance" in thestring:
#  return list(map(extract_parameter,("$0 copay",thestring)))
# if "$" in thestring and "%" in thestring:
#  return extract_copay_coinsurance(join_copay_coinsurance(list(thestring.split("and"))))
# if "$" in thestring and not ("%" in thestring):
#  return extract_copay_coinsurance("%s copay"%thestring)
# if "%" in thestring and not ("$" in thestring):
#  return extract_copay_coinsurance("%s coinsurance"%thestring)
#  return list(map(extract_parameter,("copay $0",thestring)))
