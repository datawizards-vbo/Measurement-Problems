def AB_Test(dataframe, group, target):

    # Necessary packages
    from scipy.stats import shapiro
    import scipy.stats as stats

    # # Split A/B
    control = dataframe[dataframe[group] == "Control"][target] #Old Design
    test = dataframe[dataframe[group] == "Test"][target] #New Desing

    # Assumption of the Normality 
    normality_control = shapiro(control)[1] < 0.05
    normality_test = shapiro(test)[1] < 0.05

    # H0: Data follow a normal distribution.- False
    # H1: Data do not follow a normal distribution. - True

    if (normality_control == False) & (normality_test == False):  # "H0: Data follow a normal distribution
        # Parametric Test
        # Assumption: Homogeneity of variances

        leveneTest = stats.levene(control, test)[1] < 0.05
        # H0: Homogeneity: False
        # H1: Heterogeneous: True

        if leveneTest == False:
            # Homogeneity
            ttest = stats.ttest_ind(control, test, equal_var=True)[1] # Attention! equal_var=True
            # H0: M1 == M2 - False
            # H1: M1 != M2 - True
        else:
            # Heterogeneous
            ttest = stats.ttest_ind(control, test, equal_var=False)[1] #Attention! equal_var=False
            # H0: M1 == M2 - False
            # H1: M1 != M2 - True
    else:
        # Non-Parametric Test
        ttest = stats.mannwhitneyu(control, test)[1]
        # H0: M1 == M2 - False
        # H1: M1 != M2 - True

    # Result
    temp = pd.DataFrame({
        "AB Hypothesis": [ttest < 0.05],
        "p-value": [ttest]
    })
    temp["Test Type"] = np.where((normality_control == False) & (normality_test == False), "Parametric", "Non-Parametric")
    temp["AB Hypothesis"] = np.where(temp["AB Hypothesis"] == False, "Fail to Reject H0", "Reject H0")
    temp["Comment"] = np.where(temp["AB Hypothesis"] == "Fail to Reject H0", "A/B groups are similar!",
                               "A/B groups are not similar!")

    # Columns
    if (normality_control == False) & (normality_test == False):
        temp["Homogeneity"] = np.where(leveneTest == False, "Yes", "No")
        temp = temp[["Test Type", "Homogeneity", "AB Hypothesis", "p-value", "Comment"]]
    else:
        temp = temp[["Test Type", "AB Hypothesis", "p-value", "Comment"]]

    # Print Hypothesis
    print("# A/B Testing Hypothesis")
    print("H0: A == B")
    print("H1: A != B", "\n")

    return temp