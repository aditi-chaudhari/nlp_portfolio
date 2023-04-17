from bs4 import BeautifulSoup
import re
import requests
import os
import lxml


def crawl(url):
    star_signs = ['aries', 'taurus', 'gemini', 'cancer', 'leo', 'virgo', 'libra', 'scorpio', 'sagittarius', 'capricorn',
                  'aquarius', 'pisces']

    # list of links to visit
    to_visit = {}

    # regex to validate websites
    regex = ("((http|https)://)(www.)?" +
             "[a-zA-Z0-9@:%._\\+~#?&//=]" +
             "{2,256}\\.[a-z]" +
             "{2,6}\\b([-a-zA-Z0-9@:%" +
             "._\\+~#?&//=]*)")

    # requests access to the site
    f = requests.get(url)

    # beautiful soup creates a soup object of the page
    soup = BeautifulSoup(f.content, 'lxml')

    for link in soup.find_all('a'):
        link = str(link.get('href'))
        for sign in star_signs:
            # ensures that no repeat links are returned
            if link in to_visit:
                continue

            # ensures that only the links giving general info on the zodiac signs are returned
            if 'horoscope' in link:
                continue
            if 'compatibility' in link:
                continue
            if 'article' in link:
                continue

            # validates regex
            if sign in link:
                match = re.search(regex, link)
                if match:
                    to_visit[sign] = link

    return to_visit


def scrape(links):
    for key in links:
        # using requests to access the page
        f = requests.get(links[key], headers={"User-Agent": "XY"})

        # beautiful soup creates a soup object of the page
        soup = BeautifulSoup(f.content, 'lxml')

        # extracting text from the website
        text = soup.get_text()

        # dumps all the text from the site into its own file
        with open(os.path.join(os.getcwd(), "scraped_text", f'{key}_data.txt'), mode='w') as f:
            print("Writing file...", key)
            f.write(text)
            print("Finished writing file", key)


def clean():
    directory = 'scraped_text'
    for filename in os.listdir(directory):
        # by creating a temp file, we can clean the text in place
        r = os.path.join(directory, filename)
        t = os.path.join(directory, f'tmp{filename}')

        if os.path.isfile(r):
            lines = []
            with open(r) as f, open(t, "w") as out:
                content = f.readlines()

                # i discovered that each site follows a similar structure,
                # so using this structure, i was able to find the lines
                # with the most meaningful info

                # lines 250 to 285 have general info on each sign
                general_info = content[250:285]
                for line in general_info:
                    out.write(line)

                # lines 305 to 315 have info on love and relationships for each sign
                love = content[305:315]
                for line in love:
                    out.write(line)

                # lines 400 to 430 have social info
                social = content[400:430]
                for line in social:
                    out.write(line)

        # by creating a temp file, we can clean the text in place
        os.remove(r)
        os.rename(t, r)


links_to_visit = crawl('https://www.zodiacsign.cocm')
scrape(links_to_visit)
clean()

