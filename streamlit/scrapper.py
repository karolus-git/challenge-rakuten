from bs4 import BeautifulSoup as bs
from urllib.request import urlopen

def parse(url):
    page = urlopen(url)
    soup = bs(page, "html.parser")
    return soup

def get_data_from_rakuten(soup):
    
    #Get text
    title = soup.find("span",attrs={'class': "detailHeadline"}).string
    informations = soup.find("div",attrs={'id': "prd_information"})
    texts = " ".join([info.text for info in informations])

    #Get image
    image_url = soup.find("a",attrs={'class': "prdMainPhoto"}).img["src"]
    
    return {
        "text_input" : title ,#+ " " + texts,
        "image_url_input" : image_url,
        "provider" : "Rakuten"
    }


def get_data_from_rueducommerce(soup):

    #Get text
    title = soup.find("h1",attrs={'class': "product-name"}).findAll("span")[1].text
    try:
        legend = soup.find("p",attrs={'class': "legende"}).text
    except:
        legend = ""
    informations = soup.find("div",attrs={'class': "content"})

    texts = " ".join([info.text for info in informations])

    #Get image
    gallery = soup.find("div", attrs={"class" : "gallery-box"})
    image_url_src = gallery.find("li").a.img["src"]
    image_url_input = "https://www.rueducommerce.fr" + image_url_src

    return {
        "text_input" : title + " " + legend, #+ " " + texts,
        "image_url_input" : image_url_input,
        "provider" : "RueDuCommerce"
    }

def scrap(url):
    soup = parse(url)

    if "rueducommerce" in url:
        return get_data_from_rueducommerce(soup)

    elif "rakuten" in url:
        return get_data_from_rakuten(soup)

if __name__ == "__main__":

    url = "https://www.rueducommerce.fr/p-piscine-hors-sol-tubulaire-bestway-power-steel-oval-427x250x100-cm-purateur-a-cartouche-de-2-006-lh-bestway-2007221890-26005.html"
    
    response = scrap(url)
    print(response)