# Standard library imports
import re  # Regular expressions for text processing
import html  # HTML entity handling
import logging  # Logging functionality

# Third-party imports
from bs4 import BeautifulSoup  # HTML parsing library

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HTMLTextExtractor:
    """
    Extract text from HTML while maintaining document structure.

    This class provides functionality to extract clean, readable text from HTML content
    while preserving the document's structure, including paragraphs, headings, and lists.
    It handles special characters, removes unwanted tags, and cleans up formatting issues.
    """

    def __init__(self):
        """
        Initialize the HTML text extractor with tag sets for processing.
        """
        # Tags to completely skip/remove from the HTML
        self.skip_tags = {'script', 'style', 'meta', 'link', 'head', 'noscript'}

        # Block-level tags that should have line breaks before and after
        self.block_tags = {'p', 'div', 'section', 'article', 'header', 'footer', 'aside',
                           'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'ul', 'ol', 'li'}

    def extract_text_from_html(self, html_content: str) -> str:
        """
        Extract clean text from HTML content while maintaining document structure.

        This method:
        1. Parses the HTML using BeautifulSoup
        2. Extracts the title if present
        3. Removes unwanted tags (scripts, styles, etc.)
        4. Inserts line breaks around block elements to preserve structure
        5. Extracts and cleans the text content
        6. Combines title and body text appropriately

        Args:
            html_content (str): The HTML content to process

        Returns:
            str: Clean, structured text extracted from the HTML
        """
        try:
            # Return empty string for empty input
            if not html_content.strip():
                return ""

            # Parse HTML with BeautifulSoup
            soup = BeautifulSoup(html_content, 'html.parser')

            # Extract title if present
            title = ""
            title_tag = soup.find('title')
            if title_tag:
                title = title_tag.get_text(strip=True)

            # Use body tag if available, otherwise use the entire document
            body = soup.body if soup.body else soup

            # Remove unwanted tags (scripts, styles, etc.)
            for tag_name in self.skip_tags:
                for tag in body.find_all(tag_name):
                    tag.decompose()

            # Insert line breaks around block elements to preserve structure
            for tag in body.find_all(True):
                if tag.name in self.block_tags:
                    tag.insert_before('\n\n')
                    tag.insert_after('\n\n')
                elif tag.name == 'br':
                    tag.replace_with('\n')

            # Extract text without separators so inline elements stay in-line
            body_text = body.get_text().strip()
            body_text = self._clean_text(body_text)  # Clean up the extracted text

            # Combine title and body text if both are present
            if title and body_text:
                return f"{title}\n{body_text}".strip()
            elif title:
                return title
            else:
                return body_text.strip()

        except Exception as e:
            # Fallback to simple tag removal if parsing fails
            logger.error(f"HTML parsing failed: {e}")
            return re.sub(r'<[^>]+>', '', html_content)

    def _clean_text(self, text: str) -> str:
        """
        Clean extracted text while preserving structure.

        This method performs several text cleaning operations:
        1. Unescapes HTML entities
        2. Normalizes paragraph breaks
        3. Fixes spacing and punctuation issues
        4. Handles line joining problems
        5. Cleans up special characters

        Args:
            text (str): The raw text extracted from HTML

        Returns:
            str: Cleaned text with preserved structure
        """
        # Unescape HTML entities (e.g., &amp; -> &, &quot; -> ")
        text = html.unescape(text)

        # Replace multiple newlines with just two to denote paragraphs
        text = re.sub(r'\n{3,}', '\n\n', text)

        # Process text line by line to fix various formatting issues
        lines = text.split('\n')
        processed_lines = []

        for line in lines:
            line = line.strip()
            # Skip empty lines
            if not line:
                continue

            # Fix common spacing issues:

            # Add space between lowercase and uppercase letters (camelCase -> camel Case)
            line = re.sub(r'([a-z])([A-Z])', r'\1 \2', line)

            # Fix spacing after commas
            line = re.sub(r'([a-z]),([A-Z])', r'\1, \2', line)

            # Add space after sentence-ending punctuation
            line = re.sub(r'(\w)([.!?])([A-Z])', r'\1\2 \3', line)

            # Add space after closing parenthesis before capital letter
            line = re.sub(r'(\)\s*)([A-Z])', r'\1 \2', line)

            # Add space between letters and numbers
            line = re.sub(r'([a-z])(\d)', r'\1 \2', line)
            line = re.sub(r'(\d)([a-z])', r'\1 \2', line)

            # Add space between uppercase letters and uppercase followed by lowercase
            line = re.sub(r'([A-Z])([A-Z][a-z])', r'\1 \2', line)

            # Replace middle dot with space
            line = re.sub(r'·', ' ', line)

            # Remove spaces before punctuation
            line = re.sub(r'\s+([.!?,:;])', r'\1', line)

            # Final cleanup: normalize whitespace
            line = re.sub(r'\s+', ' ', line)
            processed_lines.append(line)

        # Join processed lines with newlines
        return '\n'.join(processed_lines)


def main():
    """
    Main function to demonstrate the HTMLTextExtractor.

    This function:
    1. Creates a sample HTML document with various elements
    2. Initializes the HTMLTextExtractor
    3. Extracts clean text from the HTML
    4. Prints the result

    The sample HTML includes:
    - Document title
    - Headings and paragraphs
    - Inline formatting (strong tags)
    - Line breaks
    - Links and images
    - Script tags (which should be removed)
    - Special characters (middle dot)
    - Concatenated words (youranswers)
    """
    # Sample HTML content for demonstration
    html_content = """<!DOCTYPE html>
<html>
<head>
    <title>Airbnb - Travel Feedback</title>
</head>
<body>
    <h1>Hi Matthew,</h1>
    <p>Thanks for using <strong>airbnb</strong>. We really appreciate you choosing airbnb for your travel plans.</p>
    <p>To help us improve, we would like to ask you a few questions about your experience so far. It will only take<span class="time">3 minutes</span>, and youranswers will help us make airbnb even better for you and other guests.</p>
    <div class="footer">
        <p>Thanks,<br/>The airbnb Team</p>
        <a href="#survey" title="Take our survay">Take the Survey</a>
        <img alt="Airbnb logo for guests" src="logo.png"/>
        <p>Who is medallia?  <a href="#unsub">Unsubscribe</a> · <a href="#privacy">Privacy Policy</a></p>
        <p>DOWNLOAD ON<br/>App Store<br/>google play</p>
    </div>
    <script>
        var airbnb_data = "somthing";
    </script>
</body>
</html>"""

    # Initialize the extractor
    extractor = HTMLTextExtractor()

    # Extract text from the HTML
    result = extractor.extract_text_from_html(html_content)

    # Print the result
    print(result)


if __name__ == "__main__":
    main()
