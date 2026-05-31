import ingestion
import retrival

def ingestion_pipeline(file):
    documents = ingestion.load_documents(file)
    chunks = ingestion.split_documents(documents)
    ingestion.create_vector_store(chunks)
        
def retrival_pipeline(query):
    retrival.retrival_pipeline(query)

print("choose \n (1) to add txt file\n (2) to ask query\n (3) to exit\n")

def start():
    print("\n"+"-"*100)
    option = input("choise : ")
    if option == "1":
        file_path = input("directory path : ")
        file_name = input("file name : ")
        ingestion_pipeline(file_name, file_path)
        start()
    elif(option == "2"):
        query = input("ask ai : ")
        print(f"answer : ",end="")
        retrival_pipeline(query)
        start()
    elif(option == "3"):
        print("Thank you :)")
    else:
        print("invalid input. try again")
        start()
start()