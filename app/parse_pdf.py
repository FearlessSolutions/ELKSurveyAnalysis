
import camelot
import PyPDF2
import json

data_path = "/data/test_pdf_2.pdf"

def extract_tables():
    tables = camelot.read_pdf(data_path, pages="17", flavor="stream")
    tables[0].to_csv("/results/results2.csv")


if __name__ == "__main__":
    with open(data_path, 'rb') as pdfFileObj:

        pdfReader = PyPDF2.PdfFileReader(pdfFileObj)

        # printing number of pages in pdf file 
        print(pdfReader.numPages) 
        
        # creating a page object 
        pageObj = pdfReader.getPage(0) 
        
        # extracting text from page 

        page_text = []
        page_text.append("page,text")
        for page_index in range(pdfReader.numPages):
            pageObj = pdfReader.getPage(page_index)
            extracted_text = pageObj.extractText().replace("\n", " ").replace(",", " ")
            page_text.append("{}, {}".format(page_index, extracted_text))

        #saving text
        with open("/results/parsed_pdf.csv", "wb") as fh:
                        
            #json_text = json.dumps(page_text)

            formatted_text = "\n".join(page_text)

            fh.write(formatted_text.encode(encoding='utf-8', errors="ignore"))#all_text.encode(encoding='utf-8', errors="ignore")) 
