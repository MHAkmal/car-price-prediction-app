import marimo

__generated_with = "0.15.0"
app = marimo.App(width="full")


@app.cell
def _(mo):
    mo.md(r"""# Import Packages""")
    return


@app.cell
def _():
    import marimo as mo
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    import plotly.express as px

    from sklearn.model_selection import train_test_split, cross_validate
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import r2_score, root_mean_squared_error, mean_absolute_error
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.pipeline import Pipeline
    return (
        ColumnTransformer,
        LinearRegression,
        OneHotEncoder,
        Pipeline,
        cross_validate,
        mean_absolute_error,
        mo,
        np,
        pd,
        plt,
        r2_score,
        root_mean_squared_error,
        sns,
        train_test_split,
    )


@app.cell
def _(pd):
    df = pd.read_csv("car-price-prediction/car-price.csv")

    df.head(10)
    return (df,)


@app.cell
def _(df):
    df.info()
    return


@app.cell
def _(df):
    df.describe()
    return


@app.cell
def _(df, pd):
    df_cat = df.select_dtypes(include='object')


    summary_df = pd.DataFrame({
        'unique_count': df_cat.nunique(),
        'unique_values': df_cat.apply(lambda col: col.unique())
    })

    print(summary_df)
    return


@app.cell
def _(mo):
    mo.md(r"""## Check Missing Value""")
    return


@app.cell
def _(df):
    df.isna().sum()
    return


@app.cell
def _(mo):
    mo.md(r"""## Check Outliers""")
    return


@app.cell
def _(df):
    df_num = df.select_dtypes(["int64", "float64"])
    col_num = df_num.columns
    col_num
    return (col_num,)


@app.cell
def _(col_num, df):
    for col in col_num:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        high_fence = q3 + 1.5*iqr
        low_fence = q1 - 1.5*iqr
        outliers = df[(df[col] < low_fence) | (df[col] > high_fence)]
        print(outliers.shape)
    return


@app.cell
def _(mo):
    mo.md(r"""outliers not erased because rows of data in the dataset are small""")
    return


@app.cell
def _(df):
    df.shape
    return


@app.cell
def _(mo):
    mo.md(r"""## Check Duplicate Data""")
    return


@app.cell
def _(df):
    duplicates = df.duplicated()
    print(duplicates)

    num_duplicates = df.duplicated().sum()
    print(f"Number of duplicate rows: {num_duplicates}")
    return


@app.cell
def _(mo):
    mo.md(r"""# Exploratory Data Analysis (EDA)""")
    return


@app.cell
def _(col_num):
    col_num
    return


@app.cell
def _(mo):
    mo.md(r"""## Univariate Analysis""")
    return


@app.cell
def _(df):
    df.describe()
    return


@app.cell
def _(mo):
    mo.md(r"""look at standard deviation""")
    return


@app.cell
def _(col_num, df, plt, sns):
    plt.figure(figsize= (15,15))
    for i in range(len(col_num)):
        plt.subplot(4,4, i+1)
        sns.histplot(x = col_num[i], data=df)
        plt.tight_layout()

    plt.show()
    return


@app.cell
def _(mo):
    mo.md(r"""some features distribution are right skewed, for example: wheelbase, carwidth, curbweight, enginesize, horsepower, highwaympg, price""")
    return


@app.cell
def _(col_num):
    pairplot_list = col_num[2:10].to_list()
    pairplot_list
    return (pairplot_list,)


@app.cell
def _(mo):
    mo.md(r"""## Multivariate Analysis""")
    return


@app.cell
def _(df, pairplot_list, sns):
    sns.pairplot(df[pairplot_list])
    return


@app.cell
def _(df, pairplot_list, sns):
    sns.heatmap(df[pairplot_list].corr(), annot=True)
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    1. Based on heatmap correlation you can do feature engineering using carlength, carwidth, and carheight
    2. Some feature have strong positive correlation for example: carlength with wheelbase, carwidth with carlength, curbweight with (wheelbase, carlength, and carwidth) 
    """
    )
    return


@app.cell
def _(mo):
    mo.md(r"""# Data Preprocessing""")
    return


@app.cell
def _(mo):
    mo.md(r"""## Feature Engineering""")
    return


@app.cell
def _(df):
    df["car_area"] = df["carlength"] * df["carwidth"]
    df["car_volume"] = df["car_area"] * df["carheight"]

    df_cleaned = df.drop(columns=['car_ID', 'CarName'])
    return (df_cleaned,)


@app.cell
def _(df_cleaned):
    df_cleaned.describe()
    return


@app.cell
def _(mo):
    mo.md(r"""# Data Modeling""")
    return


@app.cell
def _(
    ColumnTransformer,
    LinearRegression,
    OneHotEncoder,
    Pipeline,
    cross_validate,
    df_cleaned,
    mean_absolute_error,
    np,
    r2_score,
    root_mean_squared_error,
    train_test_split,
):
    def create_and_evaluate_linear_regression_model(df):
        """
        Performs feature engineering, preprocessing, training, and evaluation of a
        Linear Regression model using a scikit-learn pipeline.

        Args:
            df (pd.DataFrame): The input dataframe containing the car data.
                               It is expected to have the original 26 features.

        Returns:
            tuple: A tuple containing:
                - Pipeline: The fitted scikit-learn pipeline object.
                - pd.DataFrame: X_test data.
                - pd.Series: y_test data.
        """
        df_processed = df_cleaned.copy()

        # Train Test Split
        X = df_processed.drop(columns=["price"])
        y = df_processed["price"]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

        # Create a Scikit-learn Pipeline
        categorical_features = X_train.select_dtypes(include=['object']).columns
        numerical_features = X_train.select_dtypes(include=np.number).columns

        # preprocessor object using ColumnTransformer
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', 'passthrough', numerical_features),
                ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
            ],
            remainder='passthrough' # ensures no columns are dropped
        )

        # full pipeline
        model_pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('regressor', LinearRegression())
        ])

        # Modeling and Cross-Validation
        print("--- Cross-Validation Results ---")
        scoring_metrics = ['r2', 'neg_mean_absolute_error', 'neg_root_mean_squared_error']
        cv_results = cross_validate(model_pipeline, X_train, y_train, cv=5, scoring=scoring_metrics)

        avg_cv_mae = np.mean(-cv_results['test_neg_mean_absolute_error'])
        avg_cv_rmse = np.mean(-cv_results['test_neg_root_mean_squared_error'])
        avg_cv_r2 = np.mean(cv_results['test_r2'])

        print(f"Average CV Mean Absolute Error (MAE): {avg_cv_mae:.2f}")
        print(f"Average CV Root Mean Squared Error (RMSE): {avg_cv_rmse:.2f}")
        print(f"Average CV R2 Score: {avg_cv_r2:.4f}")

        # Final Model Evaluation
        model_pipeline.fit(X_train, y_train)

        y_pred = model_pipeline.predict(X_test)

        # final metrics
        print("\n--- Final Model Evaluation on the Test Set ---")
        final_mae = mean_absolute_error(y_test, y_pred)
        final_rmse = root_mean_squared_error(y_test, y_pred)
        final_r2 = r2_score(y_test, y_pred)

        print(f"Final Mean Absolute Error (MAE): {final_mae:.2f}")
        print(f"Final Root Mean Squared Error (RMSE): {final_rmse:.2f}")
        print(f"Final R2 Score: {final_r2:.4f}")

        print("\n--- Test Data Statistics ---")
        print(f"Test Target Mean: {np.mean(y_test):.2f}")
        print(f"Test Target Standard Deviation: {np.std(y_test):.2f}")

        return model_pipeline, y_pred, X_test, y_test
    return (create_and_evaluate_linear_regression_model,)


@app.cell
def _(create_and_evaluate_linear_regression_model, df):
    fitted_pipeline, y_pred, X_test, y_test = create_and_evaluate_linear_regression_model(df)
    return fitted_pipeline, y_pred, y_test


@app.cell
def _(mo):
    mo.md(
        r"""
    # Evaluation
    1. if MAE score is below 50% of standard deviation it means the model is accurate
    2. if RMSE score is below 50% of standard deviation it means the model is accurate
    3. R2 score is the accuracy
    """
    )
    return


@app.cell
def _():
    return


@app.cell
def _(mo):
    mo.md(r"""# Business Impact""")
    return


@app.cell
def _(y_pred, y_test):
    y_diff = y_test - y_pred
    y_diff_below_50_treshold = [x for x in y_diff if (x < 0.5 * 7412.86) and (x > -0.5 * 7412.86)]
    return y_diff, y_diff_below_50_treshold


@app.cell
def _(mo):
    mo.md(
        r"""
    y_diff: The result is a list or array containing the residuals (or errors) for each prediction.

    - If a value in y_diff is positive, it means the model's prediction was lower than the actual value. The model underestimated.
    - If a value is negative, the prediction was higher than the actual value. The model overestimated.
    - If a value is zero, the prediction was perfect.

    This line uses a list comprehension to create a new list containing only the errors that fall within a specific threshold:

    - if (x < 0.5 * 7412.86) and (x > -0.5 * 7412.86): This is the filtering condition.
    - 7412.86 is the standard deviation of the data.
    - The calculation 0.5 * 7412.86 is half of the standard deviation, which equals 3706.43.
    - So, the condition is checking if the error x is between -3706.43 and +3706.43. 
    """
    )
    return


@app.cell
def _(y_diff_below_50_treshold):
    y_diff_below_50_treshold
    return


@app.cell
def _(y_diff):
    len(y_diff)
    return


@app.cell
def _(y_diff_below_50_treshold):
    len(y_diff_below_50_treshold)
    return


@app.cell
def _():
    accurate = 38/41
    accurate
    return


@app.cell
def _(mo):
    mo.md(r"""38 out of 41 the prediction is true = 93% true prediction""")
    return


@app.cell
def _(mo):
    mo.md(r"""# Export Model""")
    return


@app.cell
def _(fitted_pipeline):
    import joblib

    # Assuming 'fitted_model' is the variable holding your trained pipeline
    # fitted_model, mse, r2 = create_and_evaluate_regression_model_pl(df_cleaned)

    joblib.dump(fitted_pipeline, './car-price-prediction/car_price_prediction_model.pkl')
    print("Model saved successfully!")
    return


if __name__ == "__main__":
    app.run()
