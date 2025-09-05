import pandas as pd
import google.generativeai as genai

genai.configure(api_key="AIzaSyBif5c4kQOeJKpo-aRNQva86h1ldss_ggE")  
model = genai.GenerativeModel("gemini-2.0-flash")

df = pd.read_csv("data/Sample_Support_Emails_Dataset.csv")

if "email_text" not in df.columns:
    raise ValueError("CSV must contain a column named 'email_text'.")

def classify_email(email):
    try:
        prompt = f"""
        You are an AI assistant. Classify the following support email into:
        - Category (Billing, Technical Issue, Account, General Inquiry, Other)
        - Urgency (High, Medium, Low)
        - Suggested Response (short text)

        Email: {email}
        """
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error: {str(e)}"

df["AI_Classification"] = df["email_text"].apply(classify_email)

df.to_csv("classified_emails.csv", index=False)
print("✅ Classification completed! Results saved in 'classified_emails.csv'")
        """
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error: {str(e)}"

# -------------------------------
# 4. Apply classification
# -------------------------------
df["AI_Classification"] = df["email_text"].apply(classify_email)

# -------------------------------
# 5. Save results
# -------------------------------
df.to_csv("classified_emails.csv", index=False)
print("✅ Classification completed! Results saved in 'classified_emails.csv'")
