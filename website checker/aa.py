import asyncio
import requests
from pyppeteer import launch
from pyppeteer_stealth import stealth
from tabulate import tabulate
from datetime import datetime
import os
import json

# === CONFIG ===
CHROME_PATH = 'C:/Program Files/Google/Chrome/Application/chrome.exe'
PAGESPEED_API_KEY = 'AIzaSyCbQDhKdp0CYvolTi-fNkBaGMVmjPCc0ro'  # Replace with your actual key
REPORT_DIR = 'reports'

# Create reports directory if it doesn't exist
os.makedirs(REPORT_DIR, exist_ok=True)

async def run_axe_core(url):
    """Run accessibility audit using axe-core"""
    try:
        browser = await launch(
            headless=True,
            args=['--no-sandbox'],
            executablePath=CHROME_PATH
        )
        page = await browser.newPage()
        await stealth(page)
        
        # Set timeout and wait until network is idle
        await page.goto(url, {'waitUntil': 'networkidle2', 'timeout': 60000})
        
        # Load axe-core
        axe_cdn = 'https://cdnjs.cloudflare.com/ajax/libs/axe-core/4.4.1/axe.min.js'
        await page.addScriptTag({'url': axe_cdn})

        # Run axe accessibility audit
        results = await page.evaluate('''async () => {
            return await axe.run(document, {
                runOnly: {
                    type: "tag",
                    values: ["wcag2a", "wcag2aa"]
                }
            });
        }''')

        await browser.close()
        return results
    except Exception as e:
        print(f"[ERROR] Accessibility audit failed: {str(e)}")
        return None

def get_lighthouse_data(url, strategy='desktop'):
    """Fetch Lighthouse data from PageSpeed Insights API"""
    print(f"\n📡 Fetching Lighthouse ({strategy}) Report...")
    
    api_url = "https://www.googleapis.com/pagespeedonline/v5/runPagespeed"
    params = {
        'url': url,
        'key': PAGESPEED_API_KEY,
        'strategy': strategy,
        'category': ['performance', 'accessibility', 'best-practices', 'seo']
    }

    try:
        response = requests.get(api_url, params=params, timeout=30)
        if response.status_code != 200:
            print(f"[ERROR] API request failed with status {response.status_code}")
            return None
        return response.json()
    except Exception as e:
        print(f"[ERROR] Failed to fetch Lighthouse data: {str(e)}")
        return None

def display_axe_results(results):
    """Display accessibility results in console"""
    if not results:
        print("❌ No accessibility results returned")
        return

    violations = results.get('violations', [])

    if not violations:
        print("✅ No accessibility violations found!")
        return

    table_data = []
    for v in violations:
        description = v.get('description', 'No description available')
        impact = v.get('impact', 'none')
        help_url = v.get('helpUrl', '')
        for node in v.get('nodes', []):
            selectors = ', '.join(node.get('target', []))
            table_data.append([
                v.get('id', 'N/A'),
                impact,
                description[:100] + '...' if len(description) > 100 else description,
                selectors[:50] + '...' if len(selectors) > 50 else selectors,
                help_url
            ])

    headers = ["Rule ID", "Impact", "Description", "Element Selector", "Help URL"]
    print("\n🔍 Accessibility Violations:")
    print(tabulate(table_data, headers=headers, tablefmt="fancy_grid"))

def display_lighthouse_summary(lighthouse_data):
    """Display Lighthouse summary in console"""
    if not lighthouse_data:
        print("❌ No Lighthouse data returned")
        return

    try:
        lh = lighthouse_data.get('lighthouseResult', {})
        categories = lh.get('categories', {})
        audits = lh.get('audits', {})

        # Display category scores
        print("\n📊 Lighthouse Scores:")
        scores = []
        for cat_id, cat_data in categories.items():
            scores.append([
                cat_data.get('title', cat_id),
                f"{cat_data.get('score', 0) * 100:.0f}",
                cat_data.get('description', '')
            ])
        print(tabulate(scores, headers=["Category", "Score", "Description"], tablefmt="github"))

        # Display core web vitals
        print("\n⏱ Core Web Vitals:")
        vitals = []
        for metric in ['first-contentful-paint', 'largest-contentful-paint', 
                      'cumulative-layout-shift', 'total-blocking-time', 'speed-index']:
            if metric in audits:
                vitals.append([
                    audits[metric].get('title', metric),
                    audits[metric].get('displayValue', 'N/A'),
                    audits[metric].get('description', '')
                ])
        print(tabulate(vitals, headers=["Metric", "Value", "Description"], tablefmt="github"))

    except Exception as e:
        print(f"[ERROR] Failed to process Lighthouse data: {str(e)}")

def generate_html_report(url, axe_results, lighthouse_data):
    """Generate comprehensive HTML report"""
    try:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        safe_url = url.replace('https://', '').replace('http://', '').replace('/', '_')[:100]
        filename = f"{REPORT_DIR}/{safe_url}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        
        # Extract data with safety checks
        lh = lighthouse_data.get('lighthouseResult', {}) if lighthouse_data else {}
        categories = lh.get('categories', {})
        audits = lh.get('audits', {})
        
        # Prepare accessibility violations
        violations_html = create_accessibility_html(axe_results)
        
        # Prepare score cards
        score_cards = create_score_cards(categories)
        
        # Prepare performance metrics
        performance_metrics = create_performance_metrics(audits)
        
        # Prepare opportunities
        opportunities = create_optimization_opportunities(audits)
        
        # Prepare SEO checks
        seo_checks = create_seo_checks(audits)
        
        # Prepare best practices
        best_practices = create_best_practices(audits)
        
        # Generate the full HTML
        html = f"""
        <!DOCTYPE html>
        <html lang="en">
        {generate_html_head(url)}
        <body>
            {generate_html_header(url, timestamp)}
            
            <div class="section">
                <h2>Performance Overview</h2>
                <div class="scores-container">
                    {score_cards}
                </div>
            </div>
            
            <div class="section">
                {performance_metrics}
            </div>
            
            <div class="section">
                <h2>Accessibility Audit</h2>
                {violations_html}
            </div>
            
            <div class="section">
                {opportunities}
            </div>
            
            <div class="section">
                <h2>Additional Recommendations</h2>
                <h3>SEO Improvements</h3>
                <table class='metrics'>
                    <tr><th>Check</th><th>Status</th><th>Details</th></tr>
                    {seo_checks}
                </table>
                
                <h3>Best Practices</h3>
                <table class='metrics'>
                    <tr><th>Check</th><th>Status</th><th>Details</th></tr>
                    {best_practices}
                </table>
            </div>
        </body>
        </html>
        """
        
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(html)
        
        print(f"\n✅ Report generated: {filename}")
        return filename
    except Exception as e:
        print(f"[ERROR] Failed to generate HTML report: {str(e)}")
        return None

def generate_html_head(url):
    """Generate HTML head section"""
    return f"""
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Website Audit Report for {url}</title>
        <style>
            body {{
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                line-height: 1.6;
                color: #333;
                max-width: 1200px;
                margin: 0 auto;
                padding: 20px;
            }}
            header {{
                background-color: #f8f9fa;
                padding: 20px;
                border-radius: 8px;
                margin-bottom: 30px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }}
            h1 {{
                color: #2c3e50;
                margin-bottom: 5px;
            }}
            .report-info {{
                color: #7f8c8d;
                font-size: 14px;
            }}
            .scores-container {{
                display: flex;
                flex-wrap: wrap;
                gap: 20px;
                margin-bottom: 30px;
            }}
            .score-card {{
                flex: 1;
                min-width: 200px;
                padding: 20px;
                border-radius: 8px;
                color: white;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }}
            .score-card h3 {{
                margin-top: 0;
            }}
            .score {{
                font-size: 36px;
                font-weight: bold;
                margin: 15px 0;
                text-align: center;
            }}
            .score-high {{ background-color: #27ae60; }}
            .score-medium {{ background-color: #f39c12; }}
            .score-low {{ background-color: #e74c3c; }}
            table {{
                width: 100%;
                border-collapse: collapse;
                margin: 20px 0;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }}
            th, td {{
                padding: 12px 15px;
                text-align: left;
                border-bottom: 1px solid #ddd;
            }}
            th {{
                background-color: #f8f9fa;
            }}
            tr:hover {{
                background-color: #f5f5f5;
            }}
            .impact-critical {{ color: #e74c3c; font-weight: bold; }}
            .impact-serious {{ color: #e67e22; font-weight: bold; }}
            .impact-moderate {{ color: #f39c12; }}
            .impact-minor {{ color: #3498db; }}
            .success {{
                background-color: #d5f5e3;
                padding: 15px;
                border-radius: 5px;
                margin: 20px 0;
            }}
            code {{
                background-color: #f8f9fa;
                padding: 2px 4px;
                border-radius: 3px;
                font-family: monospace;
            }}
            .section {{
                margin-bottom: 40px;
                padding-bottom: 20px;
                border-bottom: 1px solid #eee;
            }}
            .section:last-child {{
                border-bottom: none;
            }}
            .warning {{
                background-color: #fff3cd;
                padding: 15px;
                border-radius: 5px;
                margin: 20px 0;
            }}
        </style>
    </head>
    """

def generate_html_header(url, timestamp):
    """Generate HTML header section"""
    return f"""
    <header>
        <h1>Website Audit Report</h1>
        <div class="report-info">
            <strong>URL:</strong> {url}<br>
            <strong>Generated on:</strong> {timestamp}
        </div>
    </header>
    """

def create_accessibility_html(axe_results):
    """Generate HTML for accessibility violations"""
    if not axe_results or not axe_results.get('violations'):
        return "<div class='success'>✅ No accessibility violations found!</div>"
    
    html = "<h3>Accessibility Violations</h3><table class='violations'><tr><th>Rule ID</th><th>Impact</th><th>Description</th><th>Element Selector</th><th>Help URL</th></tr>"
    
    for v in axe_results['violations']:
        description = v.get('description', 'No description available')
        impact = v.get('impact', 'none')
        help_url = v.get('helpUrl', '')
        
        for node in v.get('nodes', []):
            selectors = ', '.join(node.get('target', []))
            html += f"""
            <tr>
                <td>{v.get('id', 'N/A')}</td>
                <td class='impact-{impact}'>{impact}</td>
                <td>{description}</td>
                <td><code>{selectors}</code></td>
                <td><a href="{help_url}" target="_blank">Documentation</a></td>
            </tr>
            """
    
    html += "</table>"
    return html

def create_score_cards(categories):
    """Generate HTML for score cards"""
    html = ""
    for cat_id, cat_data in categories.items():
        score = cat_data.get('score', 0) * 100
        score_class = "score-high" if score >= 90 else "score-medium" if score >= 50 else "score-low"
        html += f"""
        <div class='score-card {score_class}'>
            <h3>{cat_data.get('title', cat_id)}</h3>
            <div class='score'>{score:.0f}</div>
            <div class='description'>{cat_data.get('description', '')}</div>
        </div>
        """
    return html

def create_performance_metrics(audits):
    """Generate HTML for performance metrics"""
    html = """
    <h3>Performance Metrics</h3>
    <table class='metrics'>
        <tr><th>Metric</th><th>Value</th><th>Display Value</th></tr>
    """
    
    metrics = [
        'first-contentful-paint', 'largest-contentful-paint', 
        'cumulative-layout-shift', 'total-blocking-time', 
        'speed-index', 'interactive', 'max-potential-fid'
    ]
    
    for metric in metrics:
        if metric in audits:
            audit = audits[metric]
            html += f"""
            <tr>
                <td>{audit.get('title', metric)}</td>
                <td>{audit.get('numericValue', 'N/A'):.2f} {audit.get('numericUnit', '')}</td>
                <td>{audit.get('displayValue', 'N/A')}</td>
            </tr>
            """
    
    html += "</table>"
    return html

def create_optimization_opportunities(audits):
    """Generate HTML for optimization opportunities"""
    html = """
    <h3>Optimization Opportunities</h3>
    <table class='opportunities'>
        <tr><th>Opportunity</th><th>Potential Savings</th><th>Details</th></tr>
    """
    
    for audit in audits.values():
        if (audit.get('score') is not None and audit['score'] < 1 and 
            'displayValue' in audit and 'numericValue' in audit):
            savings = f"{audit['numericValue']:.2f} {audit.get('numericUnit', '')}" 
            html += f"""
            <tr>
                <td>{audit.get('title', '')}</td>
                <td>{savings}</td>
                <td>{audit.get('description', '')}</td>
            </tr>
            """
    
    html += "</table>"
    return html

def create_seo_checks(audits):
    """Generate HTML for SEO checks"""
    html = ""
    for audit_id in ['meta-description', 'document-title', 'link-text', 
                    'crawlable-anchors', 'canonical', 'font-size', 'plugins']:
        if audit_id in audits:
            audit = audits[audit_id]
            status = "✅ Pass" if audit.get('score', 0) else "❌ Fail"
            html += f"""
            <tr>
                <td>{audit.get('title', audit_id)}</td>
                <td>{status}</td>
                <td>{audit.get('description', '')}</td>
            </tr>
            """
    return html

def create_best_practices(audits):
    """Generate HTML for best practices"""
    html = ""
    for audit_id in ['uses-http2', 'uses-passive-event-listeners', 'no-document-write', 
                    'external-anchors-use-rel-noopener', 'geolocation-on-start', 
                    'doctype', 'charset']:
        if audit_id in audits:
            audit = audits[audit_id]
            status = "✅ Pass" if audit.get('score', 0) else "❌ Fail"
            html += f"""
            <tr>
                <td>{audit.get('title', audit_id)}</td>
                <td>{status}</td>
                <td>{audit.get('description', '')}</td>
            </tr>
            """
    return html

def main():
    """Main function to run the audit"""
    url = input("Enter website URL (e.g. https://example.com/): ").strip()

    if not url.startswith("http"):
        url = "https://" + url

    try:
        # Run accessibility audit
        print("\n🔍 Running Accessibility Audit...")
        axe_results = asyncio.get_event_loop().run_until_complete(run_axe_core(url))
        display_axe_results(axe_results)
        
        # Run Lighthouse audit
        lighthouse_data = get_lighthouse_data(url)
        if lighthouse_data:
            display_lighthouse_summary(lighthouse_data)
            
            # Generate HTML report
            report_path = generate_html_report(url, axe_results, lighthouse_data)
            
            if report_path:
                # Optionally open the report in browser
                if input("\nOpen report in browser? (y/n): ").lower() == 'y':
                    import webbrowser
                    webbrowser.open(f'file://{os.path.abspath(report_path)}')
        
    except Exception as e:
        print(f"[ERROR] {str(e)}")
    finally:
        print("\nAudit completed.")

if __name__ == "__main__":
    main()