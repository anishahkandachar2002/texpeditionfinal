import requests
import json
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
from datetime import datetime
import time
import re
from urllib.parse import urljoin, urlparse
import base64

# Configure Streamlit page
st.set_page_config(
    page_title="PageSpeed & GDPR Analyzer",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 2rem;
    }
    
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        margin: 0.5rem 0;
    }
    
    .score-excellent { color: #0cce6b; font-weight: bold; }
    .score-good { color: #ffa400; font-weight: bold; }
    .score-poor { color: #ff4e42; font-weight: bold; }
    
    .opportunity-card {
        background: #f8f9fa;
        border-left: 4px solid #007bff;
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 5px;
    }
    
    .diagnostic-card {
        background: #fff3cd;
        border-left: 4px solid #ffc107;
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 5px;
    }
    
    .gdpr-compliant {
        background: #d4edda;
        border-left: 4px solid #28a745;
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 5px;
    }
    
    .gdpr-violation {
        background: #f8d7da;
        border-left: 4px solid #dc3545;
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 5px;
    }
    
    .gdpr-warning {
        background: #fff3cd;
        border-left: 4px solid #ffc107;
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)

class GDPRAnalyzer:
    def __init__(self):
        self.privacy_keywords = [
            'privacy policy', 'cookie policy', 'data protection', 'gdpr',
            'personal data', 'privacy notice', 'data privacy', 'cookie notice'
        ]
        
        self.tracking_domains = [
            'google-analytics.com', 'googletagmanager.com', 'facebook.com',
            'doubleclick.net', 'googlesyndication.com', 'facebook.net',
            'twitter.com', 'linkedin.com', 'youtube.com', 'instagram.com',
            'hotjar.com', 'crazyegg.com', 'mouseflow.com', 'fullstory.com'
        ]
        
        self.required_cookie_attributes = ['secure', 'samesite']
    
    def analyze_cookies(self, url):
        """Analyze cookies for GDPR compliance"""
        try:
            session = requests.Session()
            response = session.get(url, timeout=30)
            
            cookies_analysis = {
                'total_cookies': len(session.cookies),
                'cookies': [],
                'tracking_cookies': [],
                'compliance_issues': []
            }
            
            for cookie in session.cookies:
                cookie_info = {
                    'name': cookie.name,
                    'domain': cookie.domain,
                    'secure': cookie.secure,
                    'httponly': getattr(cookie, 'httponly', False),
                    'samesite': getattr(cookie, 'samesite', None),
                    'expires': cookie.expires
                }
                
                cookies_analysis['cookies'].append(cookie_info)
                
                # Check for tracking cookies
                if any(domain in cookie.domain for domain in self.tracking_domains):
                    cookies_analysis['tracking_cookies'].append(cookie_info)
                
                # Check compliance issues
                if not cookie.secure and url.startswith('https'):
                    cookies_analysis['compliance_issues'].append(
                        f"Cookie '{cookie.name}' lacks Secure flag on HTTPS site"
                    )
                
                if not getattr(cookie, 'samesite', None):
                    cookies_analysis['compliance_issues'].append(
                        f"Cookie '{cookie.name}' lacks SameSite attribute"
                    )
            
            return cookies_analysis
            
        except Exception as e:
            return {'error': str(e)}
    
    def check_privacy_policy(self, url):
        """Check for privacy policy and related links"""
        try:
            response = requests.get(url, timeout=30)
            content = response.text.lower()
            
            privacy_analysis = {
                'has_privacy_policy': False,
                'privacy_links': [],
                'cookie_notice': False,
                'gdpr_mentions': 0
            }
            
            # Check for privacy policy links
            privacy_patterns = [
                r'href=["\']([^"\']*privacy[^"\']*)["\']',
                r'href=["\']([^"\']*cookie[^"\']*)["\']',
                r'href=["\']([^"\']*data.protection[^"\']*)["\']'
            ]
            
            for pattern in privacy_patterns:
                matches = re.findall(pattern, content)
                for match in matches:
                    full_url = urljoin(url, match)
                    privacy_analysis['privacy_links'].append(full_url)
            
            # Check for privacy-related text
            for keyword in self.privacy_keywords:
                if keyword in content:
                    privacy_analysis['has_privacy_policy'] = True
                    break
            
            # Check for cookie notice/banner
            cookie_indicators = ['cookie', 'cookies', 'cookie consent', 'accept cookies']
            for indicator in cookie_indicators:
                if indicator in content:
                    privacy_analysis['cookie_notice'] = True
                    break
            
            # Count GDPR mentions
            privacy_analysis['gdpr_mentions'] = content.count('gdpr')
            
            return privacy_analysis
            
        except Exception as e:
            return {'error': str(e)}
    
    def check_third_party_scripts(self, url):
        """Analyze third-party scripts and tracking"""
        try:
            response = requests.get(url, timeout=30)
            content = response.text
            
            script_analysis = {
                'total_scripts': 0,
                'external_scripts': [],
                'tracking_scripts': [],
                'analytics_found': False,
                'social_plugins': []
            }
            
            # Find all script tags
            script_pattern = r'<script[^>]*src=["\']([^"\']+)["\'][^>]*>'
            script_matches = re.findall(script_pattern, content, re.IGNORECASE)
            
            script_analysis['total_scripts'] = len(script_matches)
            
            for script_src in script_matches:
                parsed_url = urlparse(script_src)
                domain = parsed_url.netloc
                
                if domain and domain not in urlparse(url).netloc:
                    script_analysis['external_scripts'].append(script_src)
                    
                    # Check for tracking domains
                    if any(tracking_domain in domain for tracking_domain in self.tracking_domains):
                        script_analysis['tracking_scripts'].append(script_src)
                        
                        if 'analytics' in domain:
                            script_analysis['analytics_found'] = True
                        
                        if any(social in domain for social in ['facebook', 'twitter', 'linkedin']):
                            script_analysis['social_plugins'].append(script_src)
            
            return script_analysis
            
        except Exception as e:
            return {'error': str(e)}
    
    def generate_gdpr_score(self, cookies_analysis, privacy_analysis, scripts_analysis):
        """Generate a GDPR compliance score"""
        score = 100
        issues = []
        
        # Cookie compliance (40% of score)
        if cookies_analysis.get('total_cookies', 0) > 0:
            if not privacy_analysis.get('cookie_notice', False):
                score -= 20
                issues.append("No cookie consent notice found")
            
            if cookies_analysis.get('compliance_issues'):
                score -= len(cookies_analysis['compliance_issues']) * 2
                issues.extend(cookies_analysis['compliance_issues'])
            
            if cookies_analysis.get('tracking_cookies'):
                if not privacy_analysis.get('cookie_notice', False):
                    score -= 10
                    issues.append("Tracking cookies without consent mechanism")
        
        # Privacy policy compliance (30% of score)
        if not privacy_analysis.get('has_privacy_policy', False):
            score -= 25
            issues.append("No privacy policy found")
        
        if not privacy_analysis.get('privacy_links'):
            score -= 10
            issues.append("No privacy policy links found")
        
        # Third-party scripts (30% of score)
        if scripts_analysis.get('tracking_scripts'):
            if not privacy_analysis.get('cookie_notice', False):
                score -= 15
                issues.append("Third-party tracking without consent")
        
        return max(0, score), issues

class PageSpeedAnalyzer:
    def __init__(self, api_key=None):
        self.api_key = api_key
        self.base_url = "https://www.googleapis.com/pagespeedonline/v5/runPagespeed"
    
    def analyze_url(self, url, strategy='mobile', categories=None):
        """Analyze URL using PageSpeed Insights API"""
        if categories is None:
            categories = ['performance', 'accessibility', 'best-practices', 'seo']
        
        params = {
            'url': url,
            'strategy': strategy,
            'category': categories
        }
        
        if self.api_key:
            params['key'] = self.api_key
        
        try:
            response = requests.get(self.base_url, params=params, timeout=60)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            st.error(f"Error fetching data: {str(e)}")
            return None
    
    def get_score_color(self, score):
        """Get color based on score"""
        if score >= 90:
            return "#0cce6b"
        elif score >= 50:
            return "#ffa400"
        else:
            return "#ff4e42"
    
    def get_score_class(self, score):
        """Get CSS class based on score"""
        if score >= 90:
            return "score-excellent"
        elif score >= 50:
            return "score-good"
        else:
            return "score-poor"

def create_performance_gauge(score, title):
    """Create a gauge chart for performance scores"""
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = score,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': title, 'font': {'size': 20}},
        delta = {'reference': 90},
        gauge = {
            'axis': {'range': [None, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 50], 'color': "lightgray"},
                {'range': [50, 90], 'color': "yellow"},
                {'range': [90, 100], 'color': "lightgreen"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))
    
    fig.update_layout(height=300, margin=dict(l=20, r=20, t=40, b=20))
    return fig

def create_gdpr_dashboard(gdpr_score, cookies_count, tracking_scripts_count, privacy_policy_found):
    """Create GDPR compliance dashboard"""
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('GDPR Score', 'Cookies Found', 'Tracking Scripts', 'Privacy Policy'),
        specs=[[{"type": "indicator"}, {"type": "indicator"}],
               [{"type": "indicator"}, {"type": "indicator"}]]
    )
    
    # GDPR Score gauge
    fig.add_trace(go.Indicator(
        mode="gauge+number",
        value=gdpr_score,
        title={'text': "GDPR Compliance"},
        gauge={
            'axis': {'range': [None, 100]},
            'bar': {'color': "darkgreen" if gdpr_score >= 80 else "orange" if gdpr_score >= 60 else "red"},
            'steps': [
                {'range': [0, 60], 'color': "lightgray"},
                {'range': [60, 80], 'color': "yellow"},
                {'range': [80, 100], 'color': "lightgreen"}
            ]
        }
    ), row=1, col=1)
    
    # Cookies count
    fig.add_trace(go.Indicator(
        mode="number",
        value=cookies_count,
        title={'text': "Total Cookies"},
        number={'font': {'color': "red" if cookies_count > 10 else "orange" if cookies_count > 5 else "green"}}
    ), row=1, col=2)
    
    # Tracking scripts
    fig.add_trace(go.Indicator(
        mode="number",
        value=tracking_scripts_count,
        title={'text': "Tracking Scripts"},
        number={'font': {'color': "red" if tracking_scripts_count > 3 else "orange" if tracking_scripts_count > 1 else "green"}}
    ), row=2, col=1)
    
    # Privacy policy status
    fig.add_trace(go.Indicator(
        mode="number",
        value=1 if privacy_policy_found else 0,
        title={'text': "Privacy Policy"},
        number={'font': {'color': "green" if privacy_policy_found else "red"}}
    ), row=2, col=2)
    
    fig.update_layout(height=500)
    return fig

def display_gdpr_analysis(gdpr_analyzer, url):
    """Display comprehensive GDPR analysis"""
    st.subheader("🔒 GDPR Compliance Analysis")
    
    with st.spinner("Analyzing GDPR compliance..."):
        # Analyze cookies
        cookies_analysis = gdpr_analyzer.analyze_cookies(url)
        
        # Check privacy policy
        privacy_analysis = gdpr_analyzer.check_privacy_policy(url)
        
        # Analyze third-party scripts
        scripts_analysis = gdpr_analyzer.check_third_party_scripts(url)
        
        # Generate GDPR score
        gdpr_score, gdpr_issues = gdpr_analyzer.generate_gdpr_score(
            cookies_analysis, privacy_analysis, scripts_analysis
        )
    
    # Display GDPR dashboard
    fig = create_gdpr_dashboard(
        gdpr_score,
        cookies_analysis.get('total_cookies', 0),
        len(scripts_analysis.get('tracking_scripts', [])),
        privacy_analysis.get('has_privacy_policy', False)
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Display detailed results
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🍪 Cookie Analysis")
        
        if cookies_analysis.get('error'):
            st.error(f"Error analyzing cookies: {cookies_analysis['error']}")
        else:
            total_cookies = cookies_analysis.get('total_cookies', 0)
            tracking_cookies = len(cookies_analysis.get('tracking_cookies', []))
            
            st.metric("Total Cookies", total_cookies)
            st.metric("Tracking Cookies", tracking_cookies)
            
            if cookies_analysis.get('compliance_issues'):
                st.markdown("**⚠️ Cookie Compliance Issues:**")
                for issue in cookies_analysis['compliance_issues']:
                    st.markdown(f"• {issue}")
            else:
                st.success("✅ No cookie compliance issues found")
            
            # Display cookie details
            if cookies_analysis.get('cookies'):
                with st.expander("View Cookie Details"):
                    cookies_df = pd.DataFrame(cookies_analysis['cookies'])
                    st.dataframe(cookies_df)
    
    with col2:
        st.subheader("📄 Privacy Policy Analysis")
        
        if privacy_analysis.get('error'):
            st.error(f"Error analyzing privacy policy: {privacy_analysis['error']}")
        else:
            has_policy = privacy_analysis.get('has_privacy_policy', False)
            has_cookie_notice = privacy_analysis.get('cookie_notice', False)
            policy_links = privacy_analysis.get('privacy_links', [])
            
            st.metric("Privacy Policy Found", "Yes" if has_policy else "No")
            st.metric("Cookie Notice", "Yes" if has_cookie_notice else "No")
            st.metric("GDPR Mentions", privacy_analysis.get('gdpr_mentions', 0))
            
            if policy_links:
                st.markdown("**🔗 Privacy-related Links:**")
                for link in policy_links[:5]:  # Show first 5 links
                    st.markdown(f"• [{link}]({link})")
    
    # Third-party scripts analysis
    st.subheader("📡 Third-party Scripts Analysis")
    
    if scripts_analysis.get('error'):
        st.error(f"Error analyzing scripts: {scripts_analysis['error']}")
    else:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("External Scripts", len(scripts_analysis.get('external_scripts', [])))
        with col2:
            st.metric("Tracking Scripts", len(scripts_analysis.get('tracking_scripts', [])))
        with col3:
            st.metric("Social Plugins", len(scripts_analysis.get('social_plugins', [])))
        
        if scripts_analysis.get('tracking_scripts'):
            with st.expander("🔍 View Tracking Scripts"):
                for script in scripts_analysis['tracking_scripts']:
                    st.code(script, language='url')
    
    # GDPR Recommendations
    st.subheader("💡 GDPR Compliance Recommendations")
    
    if gdpr_score >= 80:
        st.success(f"🎉 Excellent GDPR compliance! Score: {gdpr_score}/100")
    elif gdpr_score >= 60:
        st.warning(f"⚠️ Good GDPR compliance with room for improvement. Score: {gdpr_score}/100")
    else:
        st.error(f"❌ Poor GDPR compliance. Immediate action needed. Score: {gdpr_score}/100")
    
    if gdpr_issues:
        st.markdown("**🚨 Issues to Address:**")
        for issue in gdpr_issues:
            st.markdown(f"• {issue}")
    
    # Provide recommendations
    recommendations = []
    
    if not privacy_analysis.get('has_privacy_policy'):
        recommendations.append("Create and prominently display a comprehensive privacy policy")
    
    if not privacy_analysis.get('cookie_notice'):
        recommendations.append("Implement a cookie consent banner/notice")
    
    if cookies_analysis.get('tracking_cookies') and not privacy_analysis.get('cookie_notice'):
        recommendations.append("Obtain explicit consent before setting tracking cookies")
    
    if scripts_analysis.get('tracking_scripts'):
        recommendations.append("Review third-party tracking scripts and ensure proper consent")
    
    recommendations.extend([
        "Implement cookie categorization (necessary, analytics, marketing)",
        "Provide easy opt-out mechanisms for data processing",
        "Ensure data retention policies are clearly stated",
        "Implement user data access and deletion rights",
        "Consider implementing a privacy-by-design approach"
    ])
    
    if recommendations:
        st.markdown("**📋 General GDPR Recommendations:**")
        for rec in recommendations:
            st.markdown(f"• {rec}")
    
    return gdpr_score, gdpr_issues

def create_metrics_chart(core_web_vitals):
    """Create chart for Core Web Vitals"""
    metrics = []
    values = []
    colors = []
    
    for metric, data in core_web_vitals.items():
        metrics.append(metric.upper())
        values.append(data.get('numericValue', 0))
        
        # Determine color based on metric thresholds
        score = data.get('score', 0)
        if score >= 0.9:
            colors.append('#0cce6b')
        elif score >= 0.5:
            colors.append('#ffa400')
        else:
            colors.append('#ff4e42')
    
    fig = go.Figure(data=[
        go.Bar(x=metrics, y=values, marker_color=colors)
    ])
    
    fig.update_layout(
        title="Core Web Vitals Metrics",
        xaxis_title="Metrics",
        yaxis_title="Values (ms/s)",
        height=400
    )
    
    return fig

def display_opportunities(opportunities):
    """Display performance opportunities with fixes"""
    if not opportunities:
        st.success("🎉 No major opportunities found! Your site is well optimized.")
        return
    
    st.subheader("🎯 Performance Opportunities")
    
    for opp_id, opp_data in opportunities.items():
        title = opp_data.get('title', opp_id)
        description = opp_data.get('description', 'No description available')
        score = opp_data.get('score', 0)
        savings = opp_data.get('numericValue', 0)
        
        with st.expander(f"⚡ {title} - Potential savings: {savings:.2f}s"):
            st.markdown(f"**Description:** {description}")
            
            # Display details if available
            details = opp_data.get('details', {})
            if details.get('items'):
                st.markdown("**Affected Resources:**")
                for item in details['items'][:5]:  # Show top 5
                    if 'url' in item:
                        st.markdown(f"• {item['url']}")
            
            # Provide specific fixes based on opportunity type
            fixes = get_opportunity_fixes(opp_id)
            if fixes:
                st.markdown("**💡 How to Fix:**")
                for fix in fixes:
                    st.markdown(f"• {fix}")

def get_opportunity_fixes(opp_id):
    """Get specific fixes for each opportunity type"""
    fixes_map = {
        'unused-css-rules': [
            "Remove unused CSS rules from stylesheets",
            "Use tools like PurgeCSS to eliminate dead CSS",
            "Split CSS into critical and non-critical parts",
            "Load non-critical CSS asynchronously"
        ],
        'unused-javascript': [
            "Remove unused JavaScript code",
            "Use tree shaking to eliminate dead code",
            "Split JavaScript bundles by route/feature",
            "Implement code splitting with dynamic imports"
        ],
        'render-blocking-resources': [
            "Inline critical CSS in HTML head",
            "Load non-critical CSS asynchronously",
            "Defer non-critical JavaScript",
            "Use resource hints like preload and prefetch"
        ],
        'unminified-css': [
            "Minify CSS files using tools like cssnano",
            "Enable compression on your server",
            "Use build tools to automate minification"
        ],
        'unminified-javascript': [
            "Minify JavaScript using tools like Terser",
            "Enable gzip/brotli compression",
            "Use bundlers like Webpack or Rollup for optimization"
        ],
        'offscreen-images': [
            "Implement lazy loading for images",
            "Use Intersection Observer API for custom lazy loading",
            "Add loading='lazy' attribute to img tags",
            "Use libraries like lazysizes for advanced lazy loading"
        ],
        'next-gen-formats': [
            "Convert images to WebP or AVIF format",
            "Use picture element with multiple formats",
            "Implement responsive images with srcset",
            "Use image CDN services for automatic optimization"
        ]
    }
    
    return fixes_map.get(opp_id, ["Consult PageSpeed Insights documentation for specific guidance"])

def display_diagnostics(diagnostics):
    """Display diagnostic information"""
    if not diagnostics:
        return
    
    st.subheader("🔍 Diagnostic Information")
    
    for diag_id, diag_data in diagnostics.items():
        title = diag_data.get('title', diag_id)
        description = diag_data.get('description', 'No description available')
        score = diag_data.get('score', 1)
        
        if score < 1:  # Only show items that failed or have warnings
            with st.expander(f"⚠️ {title}"):
                st.markdown(f"**Issue:** {description}")
                
                # Display details if available
                details = diag_data.get('details', {})
                if details.get('items'):
                    st.markdown("**Details:**")
                    for item in details['items'][:3]:  # Show top 3
                        if isinstance(item, dict):
                            for key, value in item.items():
                                if key not in ['node', 'source']:
                                    st.markdown(f"• **{key}:** {value}")

def main():
    st.markdown('<h1 class="main-header">🚀 PageSpeed & GDPR Analyzer</h1>', unsafe_allow_html=True)
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        api_key = st.text_input(
            "Google API Key (Optional)", 
            type="password",
            help="Get your API key from Google Cloud Console. Without it, you'll have limited requests."
        )
        
        st.markdown("---")
        
        st.subheader("📱 Analysis Options")
        strategy = st.selectbox("Device Type", ["mobile", "desktop"])
        
        categories = st.multiselect(
            "Categories to Analyze",
            ["performance", "accessibility", "best-practices", "seo"],
            default=["performance", "accessibility", "best-practices", "seo"]
        )
        
        include_gdpr = st.checkbox("Include GDPR Analysis", value=True)
        
        st.markdown("---")
        st.markdown("### 📋 About")
        st.markdown("""
        This enhanced tool analyzes web page performance, accessibility, 
        best practices, SEO, and GDPR compliance. It provides detailed 
        reports with actionable recommendations for both performance 
        optimization and privacy compliance.
        """)
    
    # Main content area
    url = st.text_input(
        "🌐 Enter URL to analyze:", 
        placeholder="https://example.com",
        help="Enter the full URL including https://"
    )
    
    if st.button("🔍 Analyze Website", type="primary"):
        if not url:
            st.error("Please enter a URL to analyze")
            return
        
        if not url.startswith(('http://', 'https://')):
            url = 'https://' + url
        
        # Initialize analyzers
        pagespeed_analyzer = PageSpeedAnalyzer(api_key)
        gdpr_analyzer = GDPRAnalyzer()
        
        # Create tabs for different analyses
        if include_gdpr:
            tab1, tab2 = st.tabs(["📈 PageSpeed Analysis", "🔒 GDPR Compliance"])
        else:
            tab1 = st.container()
        
        # PageSpeed Analysis
        with tab1 if include_gdpr else tab1:
            with st.spinner("Analyzing website performance... This may take a few moments."):
                result = pagespeed_analyzer.analyze_url(url, strategy, categories)
            
            if result:
                # Display loading completed
                st.success("✅ PageSpeed analysis completed!")
                
                # Extract key data
                lighthouse_result = result.get('lighthouseResult', {})
                categories_data = lighthouse_result.get('categories', {})
                audits = lighthouse_result.get('audits', {})
                
                # Display overall scores
                st.subheader("📊 Overall Scores")
                
                cols = st.columns(len(categories_data))
                for i, (cat_id, cat_data) in enumerate(categories_data.items()):
                    with cols[i]:
                        score = int(cat_data.get('score', 0) * 100)
                        title = cat_data.get('title', cat_id.title())
                        
                        # Create gauge chart
                        fig = create_performance_gauge(score, title)
                        st.plotly_chart(fig, use_container_width=True)
                
                # Core Web Vitals (if performance is analyzed)
                if 'performance' in categories:
                    st.subheader("⚡ Core Web Vitals")
                    
                    core_vitals = {}
                    vital_metrics = ['largest-contentful-paint', 'first-input-delay', 'cumulative-layout-shift']
                    
                    for metric in vital_metrics:
                        if metric in audits:
                            core_vitals[metric] = audits[metric]
                    
                    if core_vitals:
                        fig = create_metrics_chart(core_vitals)
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Performance opportunities
                    performance_cat = categories_data.get('performance', {})
                    audit_refs = performance_cat.get('auditRefs', [])
                    
                    opportunities = {}
                    diagnostics = {}
                    
                    for audit_ref in audit_refs:
                        audit_id = audit_ref.get('id')
                        if audit_id in audits:
                            audit_data = audits[audit_id]
                            score = audit_data.get('score')
                            
                            if audit_ref.get('group') == 'load-opportunities' and score is not None and score < 1:
                                opportunities[audit_id] = audit_data
                            elif audit_ref.get('group') == 'diagnostics' and score is not None and score < 1:
                                diagnostics[audit_id] = audit_data
                    
                    display_opportunities(opportunities)
                    display_diagnostics(diagnostics)
                
                # Accessibility Issues
                if 'accessibility' in categories:
                    st.subheader("♿ Accessibility Analysis")
                    
                    accessibility_cat = categories_data.get('accessibility', {})
                    audit_refs = accessibility_cat.get('auditRefs', [])
                    
                    failed_audits = []
                    for audit_ref in audit_refs:
                        audit_id = audit_ref.get('id')
                        if audit_id in audits:
                            audit_data = audits[audit_id]
                            score = audit_data.get('score')
                            
                            if score is not None and score < 1:
                                failed_audits.append((audit_id, audit_data))
                    
                    if failed_audits:
                        for audit_id, audit_data in failed_audits[:10]:  # Show top 10
                            title = audit_data.get('title', audit_id)
                            description = audit_data.get('description', '')
                            
                            with st.expander(f"♿ {title}"):
                                st.markdown(f"**Issue:** {description}")
                                
                                # Show affected elements
                                details = audit_data.get('details', {})
                                if details.get('items'):
                                    st.markdown("**Affected Elements:**")
                                    for item in details['items'][:5]:
                                        if isinstance(item, dict) and 'node' in item:
                                            node_info = item['node']
                                            st.code(node_info.get('snippet', ''), language='html')
                    else:
                        st.success("🎉 No accessibility issues found!")
                
                # Best Practices Issues
                if 'best-practices' in categories:
                    st.subheader("✅ Best Practices Analysis")
                    
                    best_practices_cat = categories_data.get('best-practices', {})
                    audit_refs = best_practices_cat.get('auditRefs', [])
                    
                    failed_practices = []
                    for audit_ref in audit_refs:
                        audit_id = audit_ref.get('id')
                        if audit_id in audits:
                            audit_data = audits[audit_id]
                            score = audit_data.get('score')
                            
                            if score is not None and score < 1:
                                failed_practices.append((audit_id, audit_data))
                    
                    if failed_practices:
                        for audit_id, audit_data in failed_practices:
                            title = audit_data.get('title', audit_id)
                            description = audit_data.get('description', '')
                            
                            with st.expander(f"✅ {title}"):
                                st.markdown(f"**Issue:** {description}")
                    else:
                        st.success("🎉 All best practices checks passed!")
                
                # SEO Analysis
                if 'seo' in categories:
                    st.subheader("🔍 SEO Analysis")
                    
                    seo_cat = categories_data.get('seo', {})
                    audit_refs = seo_cat.get('auditRefs', [])
                    
                    seo_issues = []
                    for audit_ref in audit_refs:
                        audit_id = audit_ref.get('id')
                        if audit_id in audits:
                            audit_data = audits[audit_id]
                            score = audit_data.get('score')
                            
                            if score is not None and score < 1:
                                seo_issues.append((audit_id, audit_data))
                    
                    if seo_issues:
                        for audit_id, audit_data in seo_issues:
                            title = audit_data.get('title', audit_id)
                            description = audit_data.get('description', '')
                            
                            with st.expander(f"🔍 {title}"):
                                st.markdown(f"**Issue:** {description}")
                    else:
                        st.success("🎉 All SEO checks passed!")
        
        # GDPR Analysis Tab
        if include_gdpr:
            with tab2:
                gdpr_score, gdpr_issues = display_gdpr_analysis(gdpr_analyzer, url)
        
        # Download comprehensive report
        st.subheader("📥 Download Comprehensive Report")
        
        report_data = {
            'url': url,
            'timestamp': datetime.now().isoformat(),
            'strategy': strategy,
            'analysis_type': 'PageSpeed + GDPR' if include_gdpr else 'PageSpeed Only'
        }
        
        # Add PageSpeed data if available
        if 'result' in locals() and result:
            categories_data = result.get('lighthouseResult', {}).get('categories', {})
            report_data['pagespeed_scores'] = {
                cat_id: int(cat_data.get('score', 0) * 100) 
                for cat_id, cat_data in categories_data.items()
            }
            if 'opportunities' in locals():
                report_data['performance_opportunities'] = list(opportunities.keys())
            if 'diagnostics' in locals():
                report_data['performance_diagnostics'] = list(diagnostics.keys())
        
        # Add GDPR data if available
        if include_gdpr and 'gdpr_score' in locals():
            report_data['gdpr_analysis'] = {
                'compliance_score': gdpr_score,
                'issues_found': gdpr_issues,
                'analysis_date': datetime.now().isoformat()
            }
        
        json_str = json.dumps(report_data, indent=2)
        st.download_button(
            label="📊 Download Complete Report (JSON)",
            data=json_str,
            file_name=f"website_analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )
        
        # Generate summary report
        if 'result' in locals() and result:
            summary_report = generate_summary_report(
                url, 
                result, 
                gdpr_score if include_gdpr and 'gdpr_score' in locals() else None,
                gdpr_issues if include_gdpr and 'gdpr_issues' in locals() else None
            )
            
            st.download_button(
                label="📋 Download Summary Report (TXT)",
                data=summary_report,
                file_name=f"website_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain"
            )

def generate_summary_report(url, pagespeed_result, gdpr_score=None, gdpr_issues=None):
    """Generate a text summary report"""
    report = f"""
WEBSITE ANALYSIS REPORT
======================

URL: {url}
Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

PAGESPEED INSIGHTS SCORES:
-------------------------
"""
    
    # Add PageSpeed scores
    categories_data = pagespeed_result.get('lighthouseResult', {}).get('categories', {})
    for cat_id, cat_data in categories_data.items():
        score = int(cat_data.get('score', 0) * 100)
        title = cat_data.get('title', cat_id.title())
        report += f"{title}: {score}/100\n"
    
    # Add GDPR analysis if available
    if gdpr_score is not None:
        report += f"""
GDPR COMPLIANCE ANALYSIS:
------------------------
Compliance Score: {gdpr_score}/100

"""
        if gdpr_issues:
            report += "Issues Found:\n"
            for issue in gdpr_issues:
                report += f"- {issue}\n"
        else:
            report += "No major GDPR compliance issues found.\n"
    
    report += f"""

RECOMMENDATIONS:
---------------
1. Review and address all identified performance opportunities
2. Fix accessibility issues to improve user experience
3. Implement SEO best practices for better search visibility
"""
    
    if gdpr_score is not None:
        report += """4. Ensure GDPR compliance by implementing proper cookie consent
5. Review privacy policy and data processing practices
6. Implement user data rights (access, deletion, portability)
"""
    
    report += f"""
Generated by PageSpeed & GDPR Analyzer
Report Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
    
    return report

if __name__ == "__main__":
    main()