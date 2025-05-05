import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import re # Import regular expressions

def scrape_huggingface_organizations():
    """
    Scrapes Hugging Face organization pages to extract name, type, followers,
    model count, verification status, and enterprise status.
    """
    base_url = "https://huggingface.co/organizations"
    organizations_data = []
    page = 1
    max_pages = 5000 # Set a reasonable upper limit or find dynamically if possible

    print("Starting scraping...")

    while page <= max_pages:
        try:
            # Construct URL for the current page
            url = f"{base_url}?p={page}"
            print(f"Scraping page: {page} - URL: {url}")

            # Send HTTP GET request
            response = requests.get(url, timeout=30) # Increased timeout
            response.raise_for_status() # Raise an exception for bad status codes (4xx or 5xx)

            # Parse the HTML content
            soup = BeautifulSoup(response.content, 'html.parser')

            # Find all organization cards on the page
            # Note: CSS selectors might change; inspect the page source for accuracy
            org_cards = soup.find_all('article', class_='overview-card-wrapper') # Adjust selector if needed

            if not org_cards:
                print(f"No organization cards found on page {page}. Assuming end of list.")
                break # Exit loop if no more organizations are found

            # Extract data from each card
            for card in org_cards:
                org_info = {'Name': None, 'Org_ID': None, 'Org_URL': None, 'Type': None, 'Followers': 0,
                            'Models': 0, 'Is_Verified': False, 'Is_Enterprise': False}

                # Extract Name
                name_tag = card.find('h4')
                if name_tag:
                    org_info['Name'] = name_tag.get_text(strip=True)
                
                # Extract Organization ID/URL identifier
                link_tag = card.find('a', href=True)
                if link_tag and 'href' in link_tag.attrs:
                    # Extract org ID from href (e.g., "/meta-llama" from "/meta-llama/...")
                    href = link_tag['href']
                    # The pattern is typically /org-id or /org-id/something
                    org_id_match = re.match(r'^/([^/]+)', href)
                    if org_id_match:
                        org_info['Org_ID'] = org_id_match.group(1)
                        # Add the full URL to the organization
                        org_info['Org_URL'] = f"https://huggingface.co/{org_info['Org_ID']}"

                # Extract Type, Followers, Models - these are often in div/span tags
                # This part is highly dependent on the HTML structure and may need frequent updates
                info_tags = card.find_all('div', class_='flex items-center') # Example selector, adjust as needed
                
                followers_text = None
                models_text = None
                type_text = None

                # Attempt to find specific elements based on common patterns
                # Look for text patterns like 'X followers', 'Y models', and known types
                all_text_in_card = card.get_text(" ", strip=True)

                # Extract Followers using regex
                followers_match = re.search(r'([\d,.]+[kKmM]?) followers', all_text_in_card)
                if followers_match:
                    followers_text = followers_match.group(1)
                    # Convert k/M notation if present
                    if 'k' in followers_text.lower():
                        org_info['Followers'] = int(float(followers_text.lower().replace('k', '')) * 1000)
                    elif 'm' in followers_text.lower():
                         org_info['Followers'] = int(float(followers_text.lower().replace('m', '')) * 1000000)
                    else:
                        org_info['Followers'] = int(followers_text.replace(',', ''))
                elif re.search(r'1 follower', all_text_in_card): # Handle singular case
                     org_info['Followers'] = 1


                # Extract Models using regex
                models_match = re.search(r'([\d,]+) models?', all_text_in_card)
                if models_match:
                    models_text = models_match.group(1)
                    org_info['Models'] = int(models_text.replace(',', ''))
                elif re.search(r'1 model', all_text_in_card): # Handle singular case
                     org_info['Models'] = 1

                # Extract Type - Look for known types
                known_types = ['company', 'university', 'non-profit', 'community', 'classroom']
                # Search for type text, often near the start or associated with specific icons/classes
                type_tag = card.find('div', class_='mr-1 text-sm') # Example selector, adjust
                if type_tag:
                     potential_type = type_tag.get_text(strip=True).lower()
                     if potential_type in known_types:
                          org_info['Type'] = potential_type
                # Fallback search in text if specific tag fails
                if not org_info['Type']:
                    for t in known_types:
                        if t in all_text_in_card.split()[:5]: # Check first few words
                             org_info['Type'] = t
                             break

                # Check for Verification Badge (often an SVG or specific class)
                # This requires inspecting the HTML for the specific element indicating verification
                # Example: Look for a specific SVG path or a class like 'verified-badge'
                verified_badge = card.find('svg', {'aria-label': 'Verified badge'}) # Adjust selector based on inspection
                if verified_badge or "Verified" in all_text_in_card: # Check text as fallback
                    org_info['Is_Verified'] = True

                # Check for Enterprise Badge (similar approach to verification)
                # Example: Look for text 'Enterprise' or a specific class/icon
                if "Enterprise" in card.get_text(" ", strip=True): # Simple text check often works
                     org_info['Is_Enterprise'] = True
                     # Often enterprise orgs don't explicitly list 'company' type, infer if needed
                     if not org_info['Type']:
                          org_info['Type'] = 'company' # Common assumption for Enterprise

                if org_info['Name']: # Only add if a name was found
                    organizations_data.append(org_info)

            # Wait before scraping the next page to be polite
            time.sleep(1) # 1-second delay
            page += 1

        except requests.exceptions.RequestException as e:
            print(f"Error fetching page {page}: {e}")
            # Optional: Implement retry logic here
            time.sleep(5) # Wait longer after an error
            # Consider breaking after multiple consecutive errors
            break
        except Exception as e:
            print(f"An error occurred while processing page {page}: {e}")
            # Optional: Log the error and continue to the next page or break
            page += 1 # Try next page even if one fails parsing, but be cautious

    print(f"Scraping finished. Total organizations found: {len(organizations_data)}")
    return pd.DataFrame(organizations_data)

# --- Main Execution ---
if __name__ == "__main__":
    df_orgs = scrape_huggingface_organizations()

    # Display the first few rows of the DataFrame
    print("\nSample of Scraped Data:")
    print(df_orgs.head())

    # Save the data to a CSV file
    try:
        df_orgs.to_csv("huggingface_organizations.csv", index=False)
        print("\nData saved to huggingface_organizations.csv")
    except Exception as e:
        print(f"\nError saving data to CSV: {e}")