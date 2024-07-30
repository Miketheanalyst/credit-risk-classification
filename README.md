# Loan Risk Prediction Project

## Introduction
Welcome to our Loan Risk Prediction Project! This project aims to help a lending company predict whether a loan applicant is likely to repay their loan or default on it (fail to repay).

## Why This Project Matters
Lending money is risky. If a company can predict who is likely to default on a loan, it can make better decisions, reduce losses, and help more people responsibly.

## The Big Picture
We used historical data from a peer-to-peer lending service to build a model that predicts loan risk. Historical data means data from past loans, including whether each loan was repaid or defaulted on.

## How We Did It
1. **Collected Data:** We gathered data on previous loans, including information like loan amount, interest rates, and whether the loan was repaid or not.
2. **Built a Model:** We used a type of computer program called a "logistic regression model" to learn from the historical data. Think of this model as a very smart calculator that finds patterns in data.
3. **Trained the Model:** We fed the historical data into the model so it could learn which factors are linked to loan defaults.
4. **Tested the Model:** We tested the model with new data (not seen by the model during training) to see how well it predicts loan outcomes.

## Results
- **Accuracy:** The model correctly predicts the loan outcome 99% of the time.
- **Healthy Loans:** For loans that were repaid (healthy loans), the model is almost perfect.
- **High-Risk Loans:** For loans that were defaulted on (high-risk loans), the model is very good but not perfect.

## Why This Model is Useful
- **Reduce Risk:** By predicting high-risk loans, the company can take steps to reduce the chances of losing money.
- **Better Decision-Making:** The company can approve more loans with confidence and decli
