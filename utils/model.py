import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_curve, auc
from sklearn.preprocessing import LabelEncoder, StandardScaler
import warnings
warnings.filterwarnings('ignore')

@st.cache_resource
def train_profitability_model(df, model_type='random_forest', test_size=0.2, random_state=42):
    """Train ML model to predict order profitability"""
    if df.empty:
        return None
    
    try:
        # Create target variable: 1 if profit > 0, 0 otherwise
        df_model = df.copy()
        df_model['Is_Profitable'] = (df_model['profit'] > 0).astype(int)
        
        # Select features for modeling
        feature_columns = [
            'sales', 'discount', 'quantity', 'shipping_cost'
        ]
        
        # Add categorical features if available
        categorical_features = []
        if 'category' in df_model.columns:
            categorical_features.append('category')
        if 'sub_category' in df_model.columns:
            categorical_features.append('sub_category')
        if 'segment' in df_model.columns:
            categorical_features.append('segment')
        if 'region' in df_model.columns:
            categorical_features.append('region')
        if 'ship_mode' in df_model.columns:
            categorical_features.append('ship_mode')
        
        # Ensure all feature columns exist
        available_features = [col for col in feature_columns if col in df_model.columns]
        available_categorical = [col for col in categorical_features if col in df_model.columns]
        
        if len(available_features) < 3:
            st.warning("Insufficient features for model training")
            return None
        
        # Prepare feature matrix
        X_numeric = df_model[available_features].fillna(0)
        
        # Encode categorical variables
        le_dict = {}
        X_categorical = pd.DataFrame()
        
        for col in available_categorical:
            le = LabelEncoder()
            try:
                encoded_values = le.fit_transform(df_model[col].astype(str))
                X_categorical[col] = encoded_values
                le_dict[col] = le
            except Exception:
                continue
        
        # Combine features
        if not X_categorical.empty:
            X = pd.concat([X_numeric.reset_index(drop=True), X_categorical.reset_index(drop=True)], axis=1)
        else:
            X = X_numeric
        
        y = df_model['Is_Profitable']
        
        # Remove any remaining missing values
        valid_indices = ~(X.isnull().any(axis=1) | y.isnull())
        X = X[valid_indices]
        y = y[valid_indices]
        
        if len(X) < 100:
            st.warning("Insufficient data for reliable model training")
            return None
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # Scale features for logistic regression
        if model_type == 'logistic_regression':
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
        else:
            scaler = None
            X_train_scaled = X_train
            X_test_scaled = X_test
        
        # Train model
        if model_type == 'logistic_regression':
            model = LogisticRegression(random_state=random_state, max_iter=1000)
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
        else:  # random_forest
            model = RandomForestClassifier(
                n_estimators=100, 
                random_state=random_state,
                max_depth=10,
                min_samples_split=20
            )
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        
        # Calculate PR-AUC
        precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
        pr_auc = auc(recall, precision)
        
        # Get feature importance
        feature_importance = get_feature_importance(model, X.columns, model_type)
        
        # Generate predictions for export
        if model_type == 'logistic_regression' and scaler is not None:
            all_predictions = model.predict_proba(scaler.transform(X))[:, 1]
        else:
            all_predictions = model.predict_proba(X)[:, 1]
        
        predictions_df = df_model[valid_indices][['Order.ID', 'Customer.ID', 'Profit']].copy()
        predictions_df['Predicted_Profitability_Prob'] = all_predictions
        predictions_df['Predicted_Is_Profitable'] = (all_predictions > 0.5).astype(int)
        
        return {
            'model': model,
            'scaler': scaler,
            'accuracy': accuracy,
            'pr_auc': pr_auc,
            'feature_importance': feature_importance,
            'feature_columns': X.columns.tolist(),
            'label_encoders': le_dict,
            'predictions': predictions_df,
            'model_type': model_type
        }
        
    except Exception as e:
        st.error(f"Error training profitability model: {str(e)}")
        return None

def get_feature_importance(model, feature_names, model_type):
    """Extract feature importance from trained model"""
    try:
        if model_type == 'random_forest':
            importance_values = model.feature_importances_
        else:  # logistic_regression
            importance_values = np.abs(model.coef_[0])
        
        # Create importance dataframe
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance_values
        }).sort_values('importance', ascending=False)
        
        return importance_df
        
    except Exception as e:
        st.error(f"Error extracting feature importance: {str(e)}")
        return pd.DataFrame()

def get_model_insights(model_results):
    """Generate business insights from model results"""
    if not model_results or model_results['feature_importance'].empty:
        return []
    
    insights = []
    
    try:
        top_features = model_results['feature_importance'].head(5)
        
        # Model performance insight
        insights.append(
            f"Model achieves **{model_results['accuracy']:.1%} accuracy** with **{model_results['pr_auc']:.3f} PR-AUC** in predicting order profitability"
        )
        
        # Top driver insights
        if len(top_features) > 0:
            top_driver = top_features.iloc[0]
            insights.append(
                f"**{top_driver['feature']}** is the strongest predictor of order profitability"
            )
        
        # Feature type insights
        numeric_features = ['Sales', 'Discount', 'Quantity', 'Shipping.Cost']
        top_numeric = top_features[top_features['feature'].isin(numeric_features)]
        top_categorical = top_features[~top_features['feature'].isin(numeric_features)]
        
        if not top_numeric.empty:
            insights.append(
                f"Key numeric driver: **{top_numeric.iloc[0]['feature']}** significantly impacts profitability predictions"
            )
        
        if not top_categorical.empty:
            insights.append(
                f"Key categorical driver: **{top_categorical.iloc[0]['feature']}** shows strong predictive power"
            )
        
        return insights[:4]
        
    except Exception as e:
        st.error(f"Error generating model insights: {str(e)}")
        return []

def get_permutation_importance_fallback(df, model_results, n_repeats=5):
    """Fallback to permutation importance when SHAP is not available"""
    if not model_results or df.empty:
        return pd.DataFrame()
    
    try:
        from sklearn.inspection import permutation_importance
        
        # Prepare data same way as training
        df_model = df.copy()
        df_model['Is_Profitable'] = (df_model['Profit'] > 0).astype(int)
        
        # Get the same features used in training
        feature_columns = model_results['feature_columns']
        available_features = [col for col in feature_columns if col in df_model.columns]
        
        X = df_model[available_features].fillna(0)
        y = df_model['Is_Profitable']
        
        # Apply same preprocessing
        if model_results['scaler']:
            X_scaled = model_results['scaler'].transform(X)
        else:
            X_scaled = X
        
        # Calculate permutation importance
        perm_importance = permutation_importance(
            model_results['model'], 
            X_scaled, 
            y, 
            n_repeats=n_repeats, 
            random_state=42,
            scoring='accuracy'
        )
        
        # Create importance dataframe
        perm_df = pd.DataFrame({
            'feature': available_features,
            'importance_mean': perm_importance.importances_mean,
            'importance_std': perm_importance.importances_std
        }).sort_values('importance_mean', ascending=False)
        
        return perm_df.head(25)  # Top 25 features as specified
        
    except ImportError:
        st.info("Permutation importance calculation requires scikit-learn 0.22+")
        return model_results['feature_importance']
    except Exception as e:
        st.error(f"Error calculating permutation importance: {str(e)}")
        return model_results['feature_importance']

def predict_new_orders(new_data, model_results):
    """Predict profitability for new orders using trained model"""
    if not model_results or new_data.empty:
        return pd.DataFrame()
    
    try:
        # Prepare features same way as training
        feature_columns = model_results['feature_columns']
        X_new = new_data[feature_columns].fillna(0)
        
        # Apply scaling if needed
        if model_results['scaler']:
            X_new_scaled = model_results['scaler'].transform(X_new)
        else:
            X_new_scaled = X_new
        
        # Make predictions
        predictions = model_results['model'].predict_proba(X_new_scaled)[:, 1]
        predicted_class = (predictions > 0.5).astype(int)
        
        # Create results dataframe
        results = new_data[['Order.ID', 'Customer.ID']].copy()
        results['Predicted_Profitability_Prob'] = predictions
        results['Predicted_Is_Profitable'] = predicted_class
        
        return results
        
    except Exception as e:
        st.error(f"Error making predictions: {str(e)}")
        return pd.DataFrame()
