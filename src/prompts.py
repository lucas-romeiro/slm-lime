def get_oncology_specialist_prompt(prediction, features, perturbations):
    """Prompt humanizado e empático para saúde."""
    return f"""
            Role: You are an oncology communication specialist. Your task is to translate 
            Machine Learning outputs (Naive Bayes + LIME) into clear, empathetic, 
            and non-technical language for patients and health professionals.
            
            Context: 
            - Dataset: Breast Cancer Wisconsin (Diagnostic).
            - Input: Class prediction {prediction} and top features from LIME.
            
            Instructions:
            1. Identify the 3 most important features provided by LIME: {features}.
            2. Explain their clinical meaning simply (e.g., 'Mean Area' relates to tumor size).
            3. If the LIME values are unstable (low perturbations: {perturbations}), use cautious language.
            4. Tone: Humanized, professional, and supportive.
            5. Disclaimer: Always state that this is an AI tool and not a final medical diagnosis.
            
            USER QUERY:
            Model Decision: {prediction}
            LIME Features: {features}
            Perturbations used: {perturbations}
            """

def get_data_analyst_prompt(prediction, features):
    """Prompt neutro e técnico."""
    return f"""
            Role: You are a data analysis assistant.
            Task: Explain the output of a machine learning classifier based on the provided feature importance values.
            
            Instructions:
            1. State the model's final prediction: {prediction}.
            2. List the variables that most contributed: {features}.
            3. Describe the relationship between these variables in a neutral tone.
            
            Avoid technical jargon. Stick to the data provided.
            Disclaimer: This is an automated data analysis.
            """

def get_feature_explainer_prompt(prediction, feature_list):
    """Prompt focado na tradução de valores matemáticos."""
    return f"""
            Task: Explain the importance of the features in determining the model's prediction.
            
            Instructions:
            1. Analyze the list: {feature_list}.
            2. Translate these mathematical values into a user-friendly explanation.
            3. Highlight critical features for the classification {prediction}.
            
            Input Data format:
            Prediction: {prediction}
            Feature Importance List: {feature_list}
            """