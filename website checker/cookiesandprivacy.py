import streamlit as st
import base64
from datetime import datetime
from jinja2 import Template
from io import BytesIO
import zipfile

# Set page config
st.set_page_config(
    page_title="Privacy Policy Generator",
    page_icon="🔒",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        text-align: center;
        color: #2c3e50;
        margin-bottom: 2rem;
    }
    .section-header {
        color: #34495e;
        border-bottom: 2px solid #3498db;
        padding-bottom: 0.5rem;
        margin-top: 2rem;
    }
    .info-box {
        background-color: #e8f4f8;
        border-left: 4px solid #3498db;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 4px;
    }
    .download-section {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 8px;
        margin-top: 2rem;
    }
</style>
""", unsafe_allow_html=True)

# Privacy Policy Template
PRIVACY_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Privacy Policy - {{ company_name }}</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
            background-color: #fff;
        }
        h1 { 
            color: #2c3e50; 
            border-bottom: 3px solid #3498db; 
            padding-bottom: 10px; 
            margin-bottom: 30px;
        }
        h2 { 
            color: #34495e; 
            margin-top: 30px; 
            margin-bottom: 15px;
        }
        h3 { 
            color: #7f8c8d; 
            margin-top: 20px;
        }
        .highlight { 
            background-color: #ecf0f1; 
            padding: 15px; 
            border-left: 4px solid #3498db; 
            margin: 20px 0;
        }
        .contact-info { 
            background-color: #f8f9fa; 
            padding: 20px; 
            border-radius: 8px; 
            margin-top: 20px;
        }
        ul { 
            padding-left: 20px; 
            margin-bottom: 15px;
        }
        li { 
            margin-bottom: 8px; 
        }
        .effective-date { 
            font-style: italic; 
            color: #7f8c8d; 
            margin-bottom: 20px;
        }
        .footer {
            margin-top: 40px;
            padding-top: 20px;
            border-top: 1px solid #eee;
            color: #7f8c8d;
            font-size: 14px;
        }
    </style>
</head>
<body>
    <h1>Privacy Policy</h1>
    <p class="effective-date">Effective Date: {{ effective_date }}</p>
    
    <div class="highlight">
        <p><strong>{{ company_name }}</strong> ({{ business_type }}) respects your privacy and is committed to protecting your personal data. This privacy policy explains how we collect, use, and protect your information.</p>
    </div>

    <h2>1. Information We Collect</h2>
    <h3>Personal Information</h3>
    <ul>
        {% for item in personal_data %}
        <li>{{ item }}</li>
        {% endfor %}
    </ul>

    <h3>Technical Information</h3>
    <ul>
        {% for item in technical_data %}
        <li>{{ item }}</li>
        {% endfor %}
    </ul>

    <h2>2. How We Use Your Information</h2>
    <ul>
        {% for purpose in data_purposes %}
        <li>{{ purpose }}</li>
        {% endfor %}
    </ul>

    <h2>3. Legal Basis for Processing</h2>
    <p>We process your personal data based on:</p>
    <ul>
        {% for basis in legal_basis %}
        <li>{{ basis }}</li>
        {% endfor %}
    </ul>

    <h2>4. Data Sharing and Third Parties</h2>
    <p>We may share your information with:</p>
    <ul>
        {% for party in third_parties %}
        <li>{{ party }}</li>
        {% endfor %}
    </ul>

    <h2>5. Data Retention</h2>
    <p>We retain your personal data for {{ retention_period }} or as required by applicable law. Different types of data may be retained for different periods based on legal requirements and business needs.</p>

    <h2>6. Your Privacy Rights</h2>
    {% if gdpr_applicable %}
    <p><strong>GDPR Rights:</strong> If you are in the EU, you have the right to:</p>
    <ul>
        <li>Access your personal data and receive a copy</li>
        <li>Rectify inaccurate or incomplete data</li>
        <li>Erase your data (right to be forgotten)</li>
        <li>Restrict processing of your data</li>
        <li>Data portability</li>
        <li>Object to processing</li>
        <li>Withdraw consent at any time</li>
    </ul>
    {% endif %}

    {% if ccpa_applicable %}
    <p><strong>CCPA Rights:</strong> If you are a California resident, you have the right to:</p>
    <ul>
        <li>Know what personal information is collected about you</li>
        <li>Delete personal information we have collected</li>
        <li>Opt-out of the sale of personal information</li>
        <li>Non-discrimination for exercising your privacy rights</li>
    </ul>
    {% endif %}

    <h2>7. Cookies and Tracking Technologies</h2>
    <p>We use cookies and similar technologies for:</p>
    <ul>
        {% for cookie_type in cookie_types %}
        <li>{{ cookie_type }}</li>
        {% endfor %}
    </ul>
    <p>You can manage your cookie preferences through your browser settings or our cookie consent manager.</p>

    <h2>8. Data Security</h2>
    <p>We implement appropriate technical and organizational measures to protect your data, including:</p>
    <ul>
        {% for measure in security_measures %}
        <li>{{ measure }}</li>
        {% endfor %}
    </ul>

    <h2>9. International Data Transfers</h2>
    {% if international_transfers %}
    <p>Your data may be transferred to and processed in countries outside your jurisdiction. We ensure adequate protection through:</p>
    <ul>
        {% for safeguard in transfer_safeguards %}
        <li>{{ safeguard }}</li>
        {% endfor %}
    </ul>
    {% else %}
    <p>We do not transfer your personal data outside your country/region.</p>
    {% endif %}

    <h2>10. Children's Privacy</h2>
    <p>{% if children_data %}We may collect data from children under 13 with verified parental consent and implement additional safeguards as required by law.{% else %}We do not knowingly collect personal data from children under 13. If we become aware that we have collected such data, we will take steps to delete it.{% endif %}</p>

    <h2>11. Changes to This Policy</h2>
    <p>We may update this privacy policy from time to time to reflect changes in our practices or legal requirements. We will notify you of any material changes by posting the updated policy on this page and updating the effective date.</p>

    <h2>12. Contact Information</h2>
    <div class="contact-info">
        <p><strong>{{ company_name }}</strong></p>
        <p>{{ contact_address }}</p>
        <p>Email: {{ contact_email }}</p>
        {% if contact_phone %}<p>Phone: {{ contact_phone }}</p>{% endif %}
        {% if dpo_contact %}<p>Data Protection Officer: {{ dpo_contact }}</p>{% endif %}
    </div>

    <div class="footer">
        <p>This privacy policy was generated on {{ generation_date }} and should be reviewed by legal counsel before implementation.</p>
    </div>
</body>
</html>
"""

# Cookie Policy Template
COOKIE_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Cookie Policy - {{ company_name }}</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
            background-color: #fff;
        }
        h1 { 
            color: #2c3e50; 
            border-bottom: 3px solid #3498db; 
            padding-bottom: 10px; 
            margin-bottom: 30px;
        }
        h2 { 
            color: #34495e; 
            margin-top: 30px; 
            margin-bottom: 15px;
        }
        h3 { 
            color: #7f8c8d; 
            margin-top: 20px;
        }
        .cookie-table {
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .cookie-table th, .cookie-table td {
            border: 1px solid #ddd;
            padding: 12px;
            text-align: left;
        }
        .cookie-table th {
            background-color: #3498db;
            color: white;
            font-weight: bold;
        }
        .cookie-table tr:nth-child(even) {
            background-color: #f9f9f9;
        }
        .highlight { 
            background-color: #ecf0f1; 
            padding: 15px; 
            border-left: 4px solid #3498db; 
            margin: 20px 0;
        }
        .effective-date { 
            font-style: italic; 
            color: #7f8c8d; 
            margin-bottom: 20px;
        }
        .footer {
            margin-top: 40px;
            padding-top: 20px;
            border-top: 1px solid #eee;
            color: #7f8c8d;
            font-size: 14px;
        }
    </style>
</head>
<body>
    <h1>Cookie Policy</h1>
    <p class="effective-date">Effective Date: {{ effective_date }}</p>
    
    <div class="highlight">
        <p>This Cookie Policy explains how <strong>{{ company_name }}</strong> uses cookies and similar technologies to enhance your browsing experience.</p>
    </div>

    <h2>1. What Are Cookies?</h2>
    <p>Cookies are small text files that are placed on your device when you visit our website. They help us provide you with a better experience by remembering your preferences and analyzing how you use our site.</p>

    <h2>2. Types of Cookies We Use</h2>
    
    {% if essential_cookies %}
    <h3>Essential Cookies</h3>
    <p>These cookies are necessary for the website to function and cannot be switched off.</p>
    <table class="cookie-table">
        <tr>
            <th>Cookie Name</th>
            <th>Purpose</th>
            <th>Duration</th>
        </tr>
        {% for cookie in essential_cookies %}
        <tr>
            <td>{{ cookie.name }}</td>
            <td>{{ cookie.purpose }}</td>
            <td>{{ cookie.duration }}</td>
        </tr>
        {% endfor %}
    </table>
    {% endif %}

    {% if functional_cookies %}
    <h3>Functional Cookies</h3>
    <p>These cookies enable enhanced functionality and personalization.</p>
    <table class="cookie-table">
        <tr>
            <th>Cookie Name</th>
            <th>Purpose</th>
            <th>Duration</th>
        </tr>
        {% for cookie in functional_cookies %}
        <tr>
            <td>{{ cookie.name }}</td>
            <td>{{ cookie.purpose }}</td>
            <td>{{ cookie.duration }}</td>
        </tr>
        {% endfor %}
    </table>
    {% endif %}

    {% if analytics_cookies %}
    <h3>Analytics Cookies</h3>
    <p>These cookies help us understand how visitors interact with our website.</p>
    <table class="cookie-table">
        <tr>
            <th>Cookie Name</th>
            <th>Purpose</th>
            <th>Duration</th>
        </tr>
        {% for cookie in analytics_cookies %}
        <tr>
            <td>{{ cookie.name }}</td>
            <td>{{ cookie.purpose }}</td>
            <td>{{ cookie.duration }}</td>
        </tr>
        {% endfor %}
    </table>
    {% endif %}

    {% if marketing_cookies %}
    <h3>Marketing Cookies</h3>
    <p>These cookies are used to track visitors and display personalized ads.</p>
    <table class="cookie-table">
        <tr>
            <th>Cookie Name</th>
            <th>Purpose</th>
            <th>Duration</th>
        </tr>
        {% for cookie in marketing_cookies %}
        <tr>
            <td>{{ cookie.name }}</td>
            <td>{{ cookie.purpose }}</td>
            <td>{{ cookie.duration }}</td>
        </tr>
        {% endfor %}
    </table>
    {% endif %}

    <h2>3. Managing Your Cookie Preferences</h2>
    <p>You can control and manage cookies in various ways:</p>
    <ul>
        <li><strong>Browser Settings:</strong> Most browsers allow you to refuse cookies or delete existing ones</li>
        <li><strong>Cookie Consent Manager:</strong> Use our cookie banner to adjust your preferences</li>
        <li><strong>Opt-out Links:</strong> Visit third-party websites to opt out of their cookies</li>
    </ul>

    <h2>4. Third-Party Cookies</h2>
    <p>We may use third-party services that set their own cookies:</p>
    <ul>
        {% for service in third_party_services %}
        <li>{{ service }}</li>
        {% endfor %}
    </ul>

    <h2>5. Updates to This Policy</h2>
    <p>We may update this cookie policy from time to time. Any changes will be posted on this page with an updated effective date.</p>

    <h2>6. Contact Us</h2>
    <p>If you have any questions about our use of cookies, please contact us at {{ contact_email }}.</p>

    <div class="footer">
        <p>This cookie policy was generated on {{ generation_date }} and should be reviewed by legal counsel before implementation.</p>
    </div>
</body>
</html>
"""

def generate_html_file(template, data):
    """Generate HTML file from template and data"""
    template_obj = Template(template)
    return template_obj.render(**data)

def main():
    st.markdown('<h1 class="main-header">🔒 Cookie & Privacy Policy Generator</h1>', unsafe_allow_html=True)
    
    st.markdown('<div class="info-box">Generate professional, legally-compliant privacy policies and cookie notices for your website. Fill out the form below to create customized documents.</div>', unsafe_allow_html=True)
    
    # Initialize session state properly
    if 'generated_policies' not in st.session_state:
        st.session_state['generated_policies'] = None
    
    # Main form
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown('<h2 class="section-header">Company Information</h2>', unsafe_allow_html=True)
        
        company_name = st.text_input("Company Name *", placeholder="Your Company Name")
        business_type = st.selectbox(
            "Business Type *",
            ["Corporation", "LLC", "Partnership", "Sole Proprietorship", "Non-profit", "Other"]
        )
        
        contact_email = st.text_input("Contact Email *", placeholder="contact@yourcompany.com")
        contact_phone = st.text_input("Contact Phone", placeholder="+1 (555) 123-4567")
        contact_address = st.text_area("Contact Address *", placeholder="123 Main St, City, State, ZIP")
        dpo_contact = st.text_input("Data Protection Officer Contact", placeholder="dpo@yourcompany.com")
        
        st.markdown('<h2 class="section-header">Data Collection</h2>', unsafe_allow_html=True)
        
        # Personal data collection
        st.subheader("Personal Information Collected")
        personal_data_options = [
            "Names and contact information",
            "Email addresses", 
            "Phone numbers",
            "Billing and shipping addresses",
            "Payment information",
            "Account credentials",
            "Profile information",
            "Communication preferences",
            "Social media profiles",
            "Government-issued IDs"
        ]
        
        personal_data = st.multiselect(
            "Select types of personal data you collect:",
            personal_data_options,
            default=["Names and contact information", "Email addresses"]
        )
        
        # Technical data collection
        st.subheader("Technical Information Collected")
        technical_data_options = [
            "IP addresses",
            "Browser type and version",
            "Operating system",
            "Device information",
            "Cookies and tracking pixels",
            "Usage data and analytics",
            "Location data",
            "Session information",
            "Referral sources",
            "Search queries"
        ]
        
        technical_data = st.multiselect(
            "Select types of technical data you collect:",
            technical_data_options,
            default=["IP addresses", "Browser type and version", "Cookies and tracking pixels"]
        )
        
        st.markdown('<h2 class="section-header">Data Usage & Legal Compliance</h2>', unsafe_allow_html=True)
        
        # Data usage purposes
        st.subheader("How You Use the Data")
        purpose_options = [
            "Providing and improving our services",
            "Processing transactions and payments",
            "Sending marketing communications",
            "Customer support and communication",
            "Analytics and performance monitoring",
            "Security and fraud prevention",
            "Legal compliance and reporting",
            "Personalizing user experience",
            "Research and development",
            "Business operations and administration"
        ]
        
        data_purposes = st.multiselect(
            "Select how you use the collected data:",
            purpose_options,
            default=["Providing and improving our services", "Customer support and communication"]
        )
        
        # Legal basis
        st.subheader("Legal Basis for Processing")
        legal_basis_options = [
            "Consent of the data subject",
            "Performance of a contract",
            "Legal obligation",
            "Vital interests protection",
            "Public task performance",
            "Legitimate business interests"
        ]
        
        legal_basis = st.multiselect(
            "Select legal basis for data processing:",
            legal_basis_options,
            default=["Consent of the data subject", "Performance of a contract"]
        )
        
        # Third parties
        st.subheader("Third-Party Data Sharing")
        third_party_options = [
            "Service providers and contractors",
            "Payment processors",
            "Analytics providers (Google Analytics, etc.)",
            "Marketing and advertising partners",
            "Cloud storage providers",
            "Customer support platforms",
            "Legal authorities when required",
            "Business partners and affiliates",
            "Third-party integrations",
            "We do not share data with third parties"
        ]
        
        third_parties = st.multiselect(
            "Select third parties you share data with:",
            third_party_options,
            default=["Service providers and contractors"]
        )
        
        # Data retention
        retention_period = st.selectbox(
            "Data Retention Period",
            ["As long as necessary for stated purposes", "1 year", "2 years", "3 years", "5 years", "7 years", "Until account deletion"]
        )
        
        # Compliance
        st.subheader("Regulatory Compliance")
        gdpr_applicable = st.checkbox("GDPR Applicable (EU users)", value=True)
        ccpa_applicable = st.checkbox("CCPA Applicable (California users)", value=True)
        
        # International transfers
        international_transfers = st.checkbox("International Data Transfers")
        transfer_safeguards = []
        if international_transfers:
            safeguard_options = [
                "Standard Contractual Clauses (SCCs)",
                "Adequacy decisions",
                "Binding Corporate Rules (BCRs)",
                "Certification schemes",
                "Approved codes of conduct"
            ]
            transfer_safeguards = st.multiselect("Transfer safeguards:", safeguard_options)
        
        # Security measures
        st.subheader("Security Measures")
        security_options = [
            "SSL/TLS encryption",
            "Access controls and authentication",
            "Regular security audits",
            "Data encryption at rest",
            "Employee training on data protection",
            "Incident response procedures",
            "Regular backups",
            "Firewall protection",
            "Intrusion detection systems",
            "Secure data disposal"
        ]
        
        security_measures = st.multiselect(
            "Select security measures you implement:",
            security_options,
            default=["SSL/TLS encryption", "Access controls and authentication"]
        )
        
        # Children's data
        children_data = st.checkbox("Collect data from children under 13")
        
        st.markdown('<h2 class="section-header">Cookie Information</h2>', unsafe_allow_html=True)
        
        # Cookie types
        cookie_type_options = [
            "Essential website functionality",
            "Performance and analytics",
            "Marketing and advertising",
            "Personalization and preferences",
            "Social media integration",
            "Security and authentication",
            "Load balancing",
            "A/B testing",
            "Chat and customer support",
            "Third-party integrations"
        ]
        
        cookie_types = st.multiselect(
            "Select cookie purposes:",
            cookie_type_options,
            default=["Essential website functionality", "Performance and analytics"]
        )
        
        # Third-party services
        st.subheader("Third-Party Services")
        third_party_services = st.multiselect(
            "Select third-party services that set cookies:",
            ["Google Analytics", "Facebook Pixel", "Google Ads", "YouTube", "Twitter", "LinkedIn", "HubSpot", "Mailchimp", "Stripe", "PayPal", "Zendesk", "Intercom", "Hotjar", "Mixpanel", "Other"]
        )
        
    with col2:
        st.markdown('<h2 class="section-header">Generate Documents</h2>', unsafe_allow_html=True)
        
        # Validation
        required_fields = [company_name, business_type, contact_email, contact_address]
        all_required_filled = all(field for field in required_fields)
        
        if not all_required_filled:
            st.warning("Please fill in all required fields marked with *")
        
        # Generate button
        if st.button("🔄 Generate Policies", disabled=not all_required_filled, type="primary"):
            current_date = datetime.now().strftime("%B %d, %Y")
            
            # Prepare data for templates
            policy_data = {
                "company_name": company_name,
                "business_type": business_type,
                "contact_email": contact_email,
                "contact_phone": contact_phone,
                "contact_address": contact_address,
                "dpo_contact": dpo_contact,
                "personal_data": personal_data,
                "technical_data": technical_data,
                "data_purposes": data_purposes,
                "legal_basis": legal_basis,
                "third_parties": third_parties,
                "retention_period": retention_period,
                "gdpr_applicable": gdpr_applicable,
                "ccpa_applicable": ccpa_applicable,
                "international_transfers": international_transfers,
                "transfer_safeguards": transfer_safeguards,
                "security_measures": security_measures,
                "children_data": children_data,
                "cookie_types": cookie_types,
                "effective_date": current_date,
                "generation_date": current_date
            }
            
            cookie_data = {
                "company_name": company_name,
                "contact_email": contact_email,
                "essential_cookies": [],
                "analytics_cookies": [],
                "marketing_cookies": [],
                "functional_cookies": [],
                "third_party_services": third_party_services,
                "effective_date": current_date,
                "generation_date": current_date
            }
            
            # Generate HTML files
            privacy_html = generate_html_file(PRIVACY_TEMPLATE, policy_data)
            cookie_html = generate_html_file(COOKIE_TEMPLATE, cookie_data)
            
            # Store in session state
            st.session_state['generated_policies'] = {
                "privacy_html": privacy_html,
                "cookie_html": cookie_html,
                "company_name": company_name
            }
            
            st.success("✅ Policies generated successfully!")
        
        # Display generated policies
        if st.session_state['generated_policies'] is not None:
            st.markdown('<div class="download-section">', unsafe_allow_html=True)
            st.markdown("### 📥 Download Options")
            
            privacy_html = st.session_state['generated_policies']["privacy_html"]
            cookie_html = st.session_state['generated_policies']["cookie_html"]
            company_name_clean = st.session_state['generated_policies']["company_name"].replace(" ", "_")
            
            # HTML Downloads
            st.markdown("**HTML Files:**")
            col1, col2 = st.columns(2)
            
            with col1:
                st.download_button(
                    label="📄 Privacy Policy (HTML)",
                    data=privacy_html,
                    file_name=f"{company_name_clean}_Privacy_Policy.html",
                    mime="text/html"
                )
            
            with col2:
                st.download_button(
                    label="🍪 Cookie Policy (HTML)",
                    data=cookie_html,
                    file_name=f"{company_name_clean}_Cookie_Policy.html",
                    mime="text/html"
                )
            
            # Combined ZIP download
            st.markdown("**Complete Package:**")
            
            # Create ZIP file
            zip_buffer = BytesIO()
            with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                zip_file.writestr(f"{company_name_clean}_Privacy_Policy.html", privacy_html)
                zip_file.writestr(f"{company_name_clean}_Cookie_Policy.html", cookie_html)
            
            zip_buffer.seek(0)
            
            st.download_button(
                label="📦 Download All Files (ZIP)",
                data=zip_buffer.getvalue(),
                file_name=f"{company_name_clean}_Legal_Documents.zip",
                mime="application/zip"
            )
            
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Preview tabs
            st.markdown("### 👁️ Preview")
            tab1, tab2 = st.tabs(["Privacy Policy", "Cookie Policy"])
            
            with tab1:
                st.components.v1.html(privacy_html, height=600, scrolling=True)
            
            with tab2:
                st.components.v1.html(cookie_html, height=600, scrolling=True)
    
    # Sidebar information
    st.sidebar.markdown("### 📋 Features")
    st.sidebar.markdown("""
    - **GDPR Compliant**: All required GDPR provisions
    - **CCPA Compliant**: California privacy law requirements
    - **Professional Design**: Clean, readable output
    - **Customizable**: Tailor to your business needs
    - **Multiple Formats**: HTML and ZIP downloads
    """)
    
    st.sidebar.markdown("### ⚖️ Legal Disclaimer")
    st.sidebar.warning("""
    **Important:** Generated documents should be reviewed by qualified legal counsel before implementation.
    """)

if __name__ == "__main__":
    main()