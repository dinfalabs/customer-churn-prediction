# 🚀 Quick Start Guide

Welcome to the Customer Churn Prediction project! This guide will help you get started quickly.

## ⚡ 5-Minute Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Train the Model
```bash
python train_model.py
```

This will:
- Load the dataset
- Prepare features
- Train models
- Save the best model
- Generate performance metrics

**Expected output:**
```
================================================================================
CUSTOMER CHURN PREDICTION - MODEL TRAINING PIPELINE
================================================================================
[1/6] Loading and cleaning dataset...
[3/6] Selecting model via 5-fold CV (ROC-AUC) on train...
...
✓ TRAINING COMPLETED SUCCESSFULLY
================================================================================
```

### 3. Run the Web App
```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`

## 📊 Project Structure

```
customer-churn-prediction/
├── app.py                    # Streamlit application
├── train_model.py            # Model training script
├── README.md                 # Full documentation
├── requirements.txt          # Python dependencies
│
├── data/                     # Dataset
│   └── WA_Fn-UseC_-_Telco_Customer_Churn.csv
│
├── notebooks/                # Jupyter notebooks
│   ├── 01_EDA.ipynb          # Exploratory data analysis
│   └── 02_Model_Training.ipynb  # Model training (uses the pipeline)
│
├── src/                      # Source code
│   ├── data_loader.py        # Data loading, cleaning, validation
│   ├── feature_engineering.py # Target separation + derived features
│   ├── pipeline.py           # The single fit-once / serve-once pipeline
│   └── config.py             # Configuration settings
│
├── models/                   # Trained artifacts
│   ├── churn_pipeline.pkl    # Serialized end-to-end pipeline
│   └── metadata.json         # Model name, metrics, sklearn version
│
└── reports/                  # Generated CSV reports
```

## 🎯 Using the Web Application

### Overview Tab
- View dataset statistics
- Understand the business problem
- See key metrics

### Churn Insights Tab
- Analyze churn patterns
- View distribution by features
- Understand key drivers

### Make Prediction Tab
1. Fill in customer information
2. Click "Predict Churn"
3. View prediction result
4. Get recommendations

### Model Details Tab
- View model performance metrics
- See feature importance
- Understand which features drive churn

### Data Overview Tab
- Explore the full dataset
- View statistics by feature
- Check data quality

## 📈 Understanding the Data

### Target Variable
- **Churn**: Yes/No indicator (0 = No Churn, 1 = Churn)
- **Churn Rate**: ~26.5% (imbalanced dataset)

### Key Features
- **Tenure**: Months as customer (0-72)
- **MonthlyCharges**: Monthly bill in dollars
- **Contract**: Month-to-month, One year, Two year
- **InternetService**: No, DSL, Fiber optic
- Other services: Online Security, Tech Support, etc.

## 🤖 Model Information

### Models Trained
1. **Logistic Regression** - Simple, interpretable
2. **Random Forest** - More complex, better performance

### Evaluation Metrics
- **Accuracy**: Overall correctness
- **Precision**: When predicting churn, how often correct
- **Recall**: Percentage of actual churners identified
- **F1-Score**: Balanced metric
- **ROC-AUC**: Probability model quality

## 🔍 Making a Prediction

The easiest way is through the web app:

1. Go to "Make Prediction" tab
2. Enter customer information:
   - Demographics (gender, age, etc.)
   - Account info (tenure, services)
   - Billing (charges, contract type)
3. Click "Predict Churn"
4. See the result and recommendations

## 📊 Key Insights

### Top Churn Drivers
1. **Contract Type** - Month-to-month = high risk
2. **Tenure** - New customers churn more
3. **Internet Service** - Fiber optic customers
4. **Monthly Charges** - Higher charges = more churn
5. **Services** - Fewer services = more churn

### Recommendations
- Focus retention on new customers
- Encourage longer contracts
- Bundle services together
- Address Fiber optic service issues

## 🛠️ Customization

### Modify Model Parameters
Edit the `CANDIDATES` dict in `train_model.py`:
```python
"Random Forest": RandomForestClassifier(
    n_estimators=300,   # more trees
    max_depth=12,       # reduce to limit overfitting
    min_samples_leaf=4,
    class_weight="balanced_subsample",
),
```

### Add Custom Features
Edit `src/feature_engineering.py`:
- Modify `engineer_features()` function
- Add new feature calculations
- Save engineered features

### Change Model Algorithms
Edit `train_model.py`:
- Replace Logistic Regression
- Add XGBoost, SVM, etc.
- Compare performance

## 🐛 Troubleshooting

### Problem: "Module not found"
```bash
# Solution: Reinstall requirements
pip install -r requirements.txt --force-reinstall
```

### Problem: "Model not found when running app"
```bash
# Solution: Train the model first
python train_model.py
```

### Problem: "Port 8501 already in use"
```bash
# Solution: Use different port
streamlit run app.py --server.port 8502
```

### Problem: Slow prediction in app
```python
# Add caching in app.py
@st.cache_resource
def load_model():
    return joblib.load('models/churn_pipeline.pkl')
```

## 📚 Next Steps

### For Beginners
1. Read the README.md completely
2. Open `notebooks/01_EDA.ipynb` to explore data
3. Run the Streamlit app and make predictions
4. Review the code comments

### For Intermediate Users
1. Open `notebooks/02_Model_Training.ipynb`
2. Understand the model training process
3. Try modifying hyperparameters
4. Train new models with different algorithms

### For Advanced Users
1. Study the source code in `src/`
2. Add new models (XGBoost, LightGBM)
3. Implement SHAP for explainability
4. Deploy to cloud (AWS, GCP, Azure)

## 🔗 Useful Links

- **GitHub**: https://github.com/dinfalabs/customer-churn-prediction
- **Kaggle Dataset**: https://www.kaggle.com/blastchar/telco-customer-churn
- **Scikit-learn Docs**: https://scikit-learn.org/
- **Streamlit Docs**: https://docs.streamlit.io/

## 📞 Getting Help

1. **Check README.md** - Full documentation
2. **Check code comments** - Functions are well-documented
3. **Open GitHub Issues** - Ask questions
4. **Email** - Contact project maintainers

## ✅ Verification Checklist

After setup, verify everything works:

- [ ] `pip install -r requirements.txt` completes without errors
- [ ] `python train_model.py` completes successfully
- [ ] Model files created in `models/` directory
- [ ] `streamlit run app.py` launches the web app
- [ ] Can make predictions in the app
- [ ] Notebooks open and run without errors

## 🎓 Learning Resources

### Understanding the Code
1. Start with `src/data_loader.py` - see how data is loaded
2. Review `src/feature_engineering.py` - understand feature preparation
3. Study `src/pipeline.py` - the end-to-end modeling pipeline
4. Read `notebooks/02_Model_Training.ipynb` - training with the pipeline

### Understanding ML Concepts
1. **Classification**: Binary prediction problem
2. **Features**: Input variables (X)
3. **Target**: Output variable (y = Churn)
4. **Train/Test**: 80/20 split for validation
5. **Metrics**: Accuracy, Precision, Recall, F1, ROC-AUC

### Understanding the Business
1. **Problem**: High churn costs money
2. **Solution**: Predict who will churn
3. **Action**: Target retention campaigns
4. **Impact**: Reduce churn, increase revenue

## 🚀 Production Considerations

When deploying to production:

1. **Data Validation**: Check data quality
2. **Model Monitoring**: Track performance over time
3. **Retraining**: Retrain with new data periodically
4. **Version Control**: Track model versions
5. **API**: Create endpoints for predictions
6. **Logging**: Log predictions and errors
7. **Security**: Protect sensitive data

## 📝 Project Ideas to Extend

1. **Multi-class Classification**: Predict churn probability instead of binary
2. **Time Series**: Predict churn trend over time
3. **Recommendation Engine**: Suggest retention actions
4. **A/B Testing**: Test retention strategies
5. **Customer Segmentation**: Group customers by risk
6. **Lifetime Value**: Predict customer value
7. **Real-time Pipeline**: Process streaming data

---

## 🎉 You're All Set!

```bash
# Start here:
python train_model.py          # Train the model
streamlit run app.py           # Run the app

# Then visit:
# http://localhost:8501
```

Happy exploring! 🔮

---

**Questions?** Check README.md or open an issue on GitHub.
