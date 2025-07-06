import streamlit as st
from datetime import datetime
import re

# Page configuration
st.set_page_config(
    page_title="Terms & Conditions Generator",
    page_icon="📋",
    layout="wide"
)

def validate_inputs(website_name, company_name):
    """Validate user inputs"""
    errors = []
    if not website_name.strip():
        errors.append("Website name is required")
    if not company_name.strip():
        errors.append("Company name is required")
    if website_name and not re.match(r'^[a-zA-Z0-9\s\-_\.]+$', website_name):
        errors.append("Website name contains invalid characters")
    return errors

def get_category_specific_clauses(category):
    """Generate category-specific clauses"""
    clauses = {
        "E-commerce": {
            "product_services": """
## Products and Services

### Product Information
All product descriptions, images, and specifications on our website are provided for informational purposes. We strive for accuracy but do not warrant that product descriptions are complete, reliable, current, or error-free.

### Pricing and Payment
- All prices are listed in the applicable currency and are subject to change without notice
- Payment is due at the time of purchase
- We accept major credit cards and other payment methods as indicated
- All transactions are processed securely

### Order Processing
- Orders are subject to acceptance and availability
- We reserve the right to refuse or cancel orders at our discretion
- Order confirmation does not guarantee product availability

### Shipping and Delivery
- Shipping times are estimates and not guaranteed
- Risk of loss passes to you upon delivery
- International shipments may be subject to customs duties and taxes
""",
            "returns": """
## Returns and Refunds

### Return Policy
- Returns must be initiated within 30 days of delivery
- Items must be in original condition and packaging
- Some items may not be eligible for return (perishables, personalized items, etc.)

### Refund Process
- Refunds will be processed to the original payment method
- Processing time may take 5-10 business days
- Shipping costs are non-refundable unless the return is due to our error
"""
        },
        
        "Subscription-based Services": {
            "subscription": """
## Subscription Services

### Subscription Terms
- Subscriptions automatically renew unless cancelled
- You may cancel your subscription at any time through your account settings
- Cancellation takes effect at the end of the current billing period
- No refunds for partial subscription periods

### Billing
- Subscription fees are billed in advance
- Price changes will be communicated 30 days in advance
- Failed payments may result in service suspension

### Service Availability
- We strive for 99.9% uptime but do not guarantee uninterrupted service
- Scheduled maintenance will be announced in advance when possible
"""
        },
        
        "Blog": {
            "content": """
## Content and Intellectual Property

### User-Generated Content
- Users may post comments and other content
- You retain ownership of your content but grant us license to use it
- We reserve the right to moderate and remove inappropriate content

### Copyright Policy
- We respect intellectual property rights
- Report copyright infringement through our designated process
- Repeat infringers will be terminated
"""
        },
        
        "Social Network": {
            "community": """
## Community Guidelines

### User Conduct
- Be respectful and civil in all interactions
- Do not post harmful, offensive, or illegal content
- Respect other users' privacy and personal information
- Do not spam or engage in harassment

### Content Moderation
- We reserve the right to remove content that violates our guidelines
- Repeat violations may result in account suspension or termination
- Appeals process available for moderation decisions

### Privacy and Safety
- Report inappropriate behavior through our reporting system
- We may share information with law enforcement when required
"""
        },
        
        "Educational": {
            "educational": """
## Educational Services

### Course Access
- Course materials are for personal, non-commercial use only
- Access may be time-limited based on your enrollment
- We reserve the right to update course content

### Certificates and Credentials
- Completion certificates are provided upon meeting requirements
- Certificates do not constitute accredited credentials unless specifically stated
- Fraudulent completion attempts will result in termination

### Student Conduct
- Maintain academic integrity in all coursework
- Respect intellectual property of instructors and fellow students
- Prohibited: sharing login credentials, cheating, plagiarism
"""
        },
        
        "Finance": {
            "financial": """
## Financial Services Disclaimer

### Investment Risk
- All investments carry risk of loss
- Past performance does not guarantee future results
- Consult with qualified financial advisors before making investment decisions

### Regulatory Compliance
- Our services comply with applicable financial regulations
- We are not a registered investment advisor unless specifically stated
- Some services may not be available in all jurisdictions

### Data Security
- Financial information is encrypted and protected
- We comply with industry security standards
- Report suspected fraud immediately
"""
        }
    }
    
    return clauses.get(category, {})

def get_jurisdiction_clauses(jurisdiction):
    """Generate jurisdiction-specific clauses"""
    clauses = {
        "USA": {
            "privacy": """
## Privacy Rights (US)

### California Consumer Privacy Act (CCPA)
If you are a California resident, you have the right to:
- Know what personal information is collected about you
- Request deletion of your personal information
- Opt-out of the sale of your personal information
- Non-discrimination for exercising your privacy rights

### Children's Privacy (COPPA)
Our services are not directed to children under 13. We do not knowingly collect personal information from children under 13.
""",
            "legal": """
## Legal (US Jurisdiction)

### Governing Law
These Terms are governed by the laws of [State], United States, without regard to conflict of law principles.

### Dispute Resolution
- Disputes will be resolved through binding arbitration
- Class action lawsuits are waived
- Small claims court remains available for qualifying disputes

### Limitation of Liability
Our liability is limited to the maximum extent permitted by law. In no event shall our liability exceed the amount paid by you for our services in the 12 months preceding the claim.
"""
        },
        
        "Europe": {
            "privacy": """
## Privacy Rights (GDPR)

### Your Rights Under GDPR
As a data subject, you have the right to:
- Access your personal data
- Rectify inaccurate personal data
- Erase your personal data ("right to be forgotten")
- Restrict processing of your personal data
- Data portability
- Object to processing
- Withdraw consent at any time

### Legal Basis for Processing
We process your data based on:
- Your consent
- Contractual necessity
- Legal obligations
- Legitimate interests (balanced against your rights)

### Data Protection Officer
Contact our Data Protection Officer at: [dpo@company.com]
""",
            "legal": """
## Legal (European Jurisdiction)

### Governing Law
These Terms are governed by the laws of the European Union and the laws of [Country].

### Dispute Resolution
- EU residents may use the European Commission's Online Dispute Resolution platform
- Local courts maintain jurisdiction for consumer disputes
- Alternative dispute resolution mechanisms are available

### Consumer Rights
EU consumer protection laws apply where applicable and supersede conflicting terms.
"""
        },
        
        "Global": {
            "privacy": """
## Privacy Rights (Global)

### International Data Transfers
- We may transfer your data internationally
- Adequate protection measures are in place
- You may object to international transfers

### Regional Privacy Laws
- We comply with applicable privacy laws in your jurisdiction
- Specific rights may vary by location
- Contact us for region-specific privacy information
""",
            "legal": """
## Legal (Global Jurisdiction)

### Governing Law
These Terms are governed by the laws of [Company's Primary Jurisdiction], except where local laws provide greater consumer protection.

### International Compliance
- We comply with applicable laws in jurisdictions where we operate
- Local laws may provide additional rights and protections
- Some services may not be available in all countries
"""
        }
    }
    
    return clauses.get(jurisdiction, {})

def generate_terms_and_conditions(website_name, company_name, category, jurisdiction, effective_date):
    """Generate complete Terms and Conditions"""
    
    category_clauses = get_category_specific_clauses(category)
    jurisdiction_clauses = get_jurisdiction_clauses(jurisdiction)
    
    tc_template = f"""# Terms and Conditions

**Effective Date:** {effective_date}

## 1. Introduction

Welcome to {website_name}, operated by {company_name} ("we," "our," or "us"). These Terms and Conditions ("Terms") govern your use of our website and services.

By accessing or using our services, you agree to be bound by these Terms. If you disagree with any part of these Terms, you may not access our services.

## 2. Definitions

- **"Service"** refers to the {website_name} website and all related services
- **"User"** or **"you"** refers to the individual accessing our Service
- **"Company"** refers to {company_name}
- **"Content"** refers to all text, graphics, images, and other materials on our Service

## 3. Acceptance of Terms

By using our Service, you represent that:
- You are at least 18 years old or have parental consent
- You have the legal capacity to enter into these Terms
- You will comply with all applicable laws and regulations

## 4. Use of Service

### Permitted Use
You may use our Service for lawful purposes only. You agree not to:
- Violate any laws or regulations
- Infringe on intellectual property rights
- Transmit harmful or malicious code
- Interfere with the Service's operation
- Attempt unauthorized access to our systems

### Account Responsibility
If you create an account:
- You are responsible for maintaining account security
- You must provide accurate and complete information
- You must promptly update any changes to your information
- You are liable for all activities under your account

{category_clauses.get('product_services', '')}
{category_clauses.get('subscription', '')}
{category_clauses.get('content', '')}
{category_clauses.get('community', '')}
{category_clauses.get('educational', '')}
{category_clauses.get('financial', '')}
{category_clauses.get('returns', '')}

## 5. Intellectual Property

### Our Content
All content on our Service, including text, graphics, logos, and software, is owned by {company_name} or our licensors and is protected by copyright and other intellectual property laws.

### User Content
By submitting content to our Service, you grant us a non-exclusive, worldwide, royalty-free license to use, modify, and distribute your content in connection with our Service.

## 6. Privacy Policy

Your privacy is important to us. Our Privacy Policy explains how we collect, use, and protect your information. By using our Service, you consent to our privacy practices.

{jurisdiction_clauses.get('privacy', '')}

## 7. Disclaimers

### Service Availability
- Our Service is provided "as is" and "as available"
- We do not guarantee uninterrupted or error-free service
- We reserve the right to modify or discontinue the Service

### Content Accuracy
- We strive for accuracy but do not warrant the completeness or reliability of our content
- User-generated content does not reflect our views or opinions

## 8. Limitation of Liability

To the maximum extent permitted by law:
- We shall not be liable for indirect, incidental, or consequential damages
- Our total liability shall not exceed the amount paid by you for our services
- Some jurisdictions do not allow liability limitations, so these may not apply to you

## 9. Indemnification

You agree to indemnify and hold harmless {company_name} from any claims, damages, or expenses arising from:
- Your use of the Service
- Your violation of these Terms
- Your infringement of third-party rights

## 10. Termination

### By You
You may terminate your account at any time by contacting us or using account settings.

### By Us
We may terminate or suspend your access immediately if you:
- Violate these Terms
- Engage in fraudulent activity
- Pose a security risk

## 11. Changes to Terms

We reserve the right to modify these Terms at any time. Changes will be effective immediately upon posting. Your continued use constitutes acceptance of the modified Terms.

We will notify users of material changes through:
- Email notification (if you have an account)
- Prominent notice on our website
- In-app notifications (where applicable)

{jurisdiction_clauses.get('legal', '')}

## 12. Contact Information

If you have questions about these Terms, please contact us:

**{company_name}**
Email: [contact@{website_name.lower().replace(' ', '')}.com]
Address: [Company Address]
Phone: [Company Phone]

## 13. Severability

If any provision of these Terms is found to be unenforceable, the remaining provisions will remain in full force and effect.

## 14. Entire Agreement

These Terms constitute the entire agreement between you and {company_name} regarding the use of our Service.

---

*Last updated: {effective_date}*
*Generated by Terms & Conditions Generator*
"""
    
    return tc_template

# Streamlit UI
def main():
    st.title("📋 Terms & Conditions Generator")
    st.markdown("Generate comprehensive Terms and Conditions tailored to your website's category and jurisdiction.")
    
    # Sidebar for inputs
    st.sidebar.header("Configuration")
    
    # Basic Information
    st.sidebar.subheader("Basic Information")
    website_name = st.sidebar.text_input("Website Name", placeholder="e.g., MyStore, BlogSite")
    company_name = st.sidebar.text_input("Company Name", placeholder="e.g., ABC Corp, John's Business LLC")
    
    # Category Selection
    st.sidebar.subheader("Website Category")
    categories = [
        "E-commerce", "Subscription-based Services", "Blog", "News", "Content Publishing",
        "Marketplace", "Aggregators", "Community", "Forum", "Social Network",
        "Educational", "LMS", "Portfolio", "Personal Website", "Mobile App",
        "Finance", "Legal", "Medical", "Consulting Services", "Entertainment", "Media Streaming"
    ]
    category = st.sidebar.selectbox("Select Category", categories)
    
    # Jurisdiction Selection
    st.sidebar.subheader("Jurisdiction & Compliance")
    jurisdictions = ["USA", "Europe", "Global"]
    jurisdiction = st.sidebar.selectbox("Select Jurisdiction", jurisdictions)
    
    # Jurisdiction info
    jurisdiction_info = {
        "USA": "Includes CCPA (California), COPPA compliance, US arbitration clauses",
        "Europe": "Includes GDPR compliance, EU consumer protection laws",
        "Global": "Generic international compliance, adaptable to local laws"
    }
    st.sidebar.info(f"**{jurisdiction}:** {jurisdiction_info[jurisdiction]}")
    
    # Effective Date
    effective_date = st.sidebar.date_input("Effective Date", datetime.now())
    
    # Generate button
    if st.sidebar.button("Generate Terms & Conditions", type="primary"):
        # Validate inputs
        errors = validate_inputs(website_name, company_name)
        
        if errors:
            st.error("Please fix the following errors:")
            for error in errors:
                st.error(f"• {error}")
        else:
            # Generate T&C
            with st.spinner("Generating Terms & Conditions..."):
                tc_content = generate_terms_and_conditions(
                    website_name, company_name, category, jurisdiction, effective_date.strftime("%B %d, %Y")
                )
            
            # Display success message
            st.success("✅ Terms & Conditions generated successfully!")
            
            # Main content area
            col1, col2 = st.columns([3, 1])
            
            with col1:
                st.subheader(f"Terms & Conditions for {website_name}")
                
                # Display in expandable sections for better readability
                with st.expander("Preview Terms & Conditions", expanded=True):
                    st.markdown(tc_content)
            
            with col2:
                st.subheader("Actions")
                
                # Download button
                st.download_button(
                    label="📥 Download as TXT",
                    data=tc_content,
                    file_name=f"{website_name.replace(' ', '_')}_terms_conditions.txt",
                    mime="text/plain"
                )
                
                # Copy to clipboard (text area for easy copying)
                st.text_area(
                    "Copy Text:",
                    value=tc_content,
                    height=200,
                    help="Select all text and copy to clipboard"
                )
                
                # Summary info
                st.info(f"""
                **Generated for:**
                - Website: {website_name}
                - Company: {company_name}
                - Category: {category}
                - Jurisdiction: {jurisdiction}
                - Date: {effective_date.strftime("%B %d, %Y")}
                """)
                
                # Word count
                word_count = len(tc_content.split())
                st.metric("Word Count", word_count)
    
    # Information section
    if not website_name or not company_name:
        st.info("👈 Please fill in the basic information in the sidebar to generate your Terms & Conditions.")
        
        # Feature highlights
        st.markdown("## 🌟 Features")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            **📝 Category-Specific Clauses**
            - E-commerce: Returns, shipping, payments
            - Social Networks: Community guidelines
            - Educational: Course access, certificates
            - Finance: Investment disclaimers, compliance
            """)
        
        with col2:
            st.markdown("""
            **🌍 Jurisdiction Compliance**
            - **USA**: CCPA, COPPA, arbitration
            - **Europe**: GDPR, consumer protection
            - **Global**: International best practices
            """)
        
        with col3:
            st.markdown("""
            **⚡ Professional Output**
            - Comprehensive legal coverage
            - Ready-to-use format
            - Downloadable text file
            - Professional structure
            """)
        
        # Disclaimer
        st.warning("""
        **⚠️ Legal Disclaimer:** This tool generates template Terms & Conditions for informational purposes only. 
        These templates should be reviewed by qualified legal counsel before use. Laws vary by jurisdiction and business type. 
        We are not responsible for the legal adequacy of generated content.
        """)

if __name__ == "__main__":
    main()