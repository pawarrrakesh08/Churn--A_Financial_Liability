import numpy as np
import pandas as pd
import os

np.random.seed(42)

N_CUSTOMERS = 5000
N_MONTHS = 24
START_DATE = '2020-01-01'


# CUSTOMER MASTER

def generate_customer_master(n_customers=N_CUSTOMERS):
    customer_id = np.arange(1,n_customers+1)

    contract_type = np.random.choice(
        ['Monthly','Quarterly','Half-Yearly','Annual'],
        size = n_customers,
        p=[0.45,0.25,0.20,0.10]
    )

    segment = np.random.choice(
        ['SMB','MidMarket','Enterprise'],
        size=n_customers,
        p = [0.5,0.35,0.15]
    )

    base_arpu = np.round(
        np.random.lognormal(mean=8.0,sigma=0.45,size=n_customers),2
    )

    tenure_months = np.random.randint(1,60,size=n_customers)


    customer_master = pd.DataFrame({
        "customer_id":customer_id,
        "contract_type": contract_type,
        "segment": segment,
        "base_arpu": base_arpu,
        "tenure_months":tenure_months
    })

    return customer_master

# MONTHLY BEHAVIOUR

def generate_monthly_behavior(customer_master, n_months=N_MONTHS):

    records = []
    months = pd.date_range(start=START_DATE, periods=n_months, freq="MS")

    for _, row in customer_master.iterrows():

        usage_level = np.random.uniform(0.6, 1.0)
        payment_delay = np.random.exponential(scale=2)
        discount = 0.0
        churn_risk_latent = np.random.beta(2, 6)

        for month in months:

            # Latent deterioration
            churn_risk_latent += np.random.normal(0, 0.03)
            churn_risk_latent = np.clip(churn_risk_latent, 0, 1)

            # Usage decay
            usage = usage_level * (1 - churn_risk_latent)
            usage = np.clip(usage, 0, 1)

            # Discounts kick in for risky customers
            if churn_risk_latent > 0.6:
                discount = np.random.uniform(0.05, 0.25)

            # Payment stress rises
            payment_delay += churn_risk_latent * np.random.uniform(0, 4)

            # Support tickets
            tickets = np.random.poisson(
                lam=1 + churn_risk_latent * 3
            )

            arpu = row["base_arpu"] * usage * (1 - discount)

            records.append([
                row["customer_id"],
                month,
                usage,
                arpu,
                discount,
                tickets,
                payment_delay
            ])

            usage_level = usage

        # reset for next customer
    monthly_behavior = pd.DataFrame(
        records,
        columns=[
            "customer_id",
            "month",
            "usage_index",
            "arpu",
            "discount",
            "support_tickets",
            "payment_delay_days"
        ]
    )

    return monthly_behavior



# CHURN EVENTS

def generate_churn_events(customer_master,monthly_behavior):

    churn_events = []

    for cid, group in monthly_behavior.groupby("customer_id"):

        churn_prob_series = (
            0.4 * (1- group["usage_index"]) +
            0.3 * (group["payment_delay_days"]/30) +
            0.3 * group["discount"]
        )

        churn_prob_series = churn_prob_series.clip(0,1)

        chrun_draw = np.random.uniform(size=len(churn_prob_series))
        churn_flags = chrun_draw < churn_prob_series

        if churn_flags.any():
            churn_month = group.loc[churn_flags.idxmax(),"month"]
            churned = 1
        else:
            churn_month = pd.NaT
            churned = 0

        churn_events.append([
            cid,
            churned,
            churn_month
        ])


    churn_events = pd.DataFrame(
        churn_events,
        columns = ['customer_id','churn','churn_month']
    )

    return churn_events


# REVENUE LOSS

def generate_revenue_loss(customer_master,churn_events,horizon_months=6):

    records = []

    for _, row in churn_events.iterrows():

        cid = row["customer_id"]
        churned = row["churn"]

        base_arpu = customer_master.loc[
            customer_master["customer_id"] == cid, "base_arpu"
        ].values[0]


        if churned:
            decay = np.random.uniform(0.6,1.3)
            loss = base_arpu * horizon_months * decay
        else:
            loss= 0.0

        records.append([cid,round(loss,2)])

    revenue_loss = pd.DataFrame(
        records,
        columns=['customer_id','revenue_loss_6m']
    )

    return revenue_loss


# PIPELINE EXECUTION

def run_pipeline():

    customer_master = generate_customer_master()
    monthly_behavior = generate_monthly_behavior(customer_master)
    churn_events = generate_churn_events(customer_master,monthly_behavior)
    revenue_loss = generate_revenue_loss(customer_master,churn_events)

    return customer_master, monthly_behavior, churn_events, revenue_loss



os.makedirs("data/raw", exist_ok=True)

customer_master, monthly_behavior, churn_events, revenue_loss = run_pipeline()

customer_master.to_csv("data/raw/customer_master.csv", index=False)
monthly_behavior.to_csv("data/raw/customer_monthly_behavior.csv", index=False)
churn_events.to_csv("data/raw/churn_events.csv", index=False)
revenue_loss.to_csv("data/raw/revenue_loss.csv", index=False)








