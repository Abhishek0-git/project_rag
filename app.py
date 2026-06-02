from playground import advanced_retrieval

def ingestion_pipeline(file_path):
    documents = advanced_retrieval.load_documents(file_path)
    chunks = advanced_retrieval.split_documents(documents)
    advanced_retrieval.create_vector_store(chunks)
        
def retrival_pipeline(query):
    advanced_retrieval.ask_query(query)

def start():
    while True:
        print("\n"+"-"*100)
        option = input("choice : ")
        if option == "1":
            file_path = input("directory path : ")
            ingestion_pipeline(file_path)
        elif option == "2":
            query = input("ask ai : ")
            print(f"answer : ", end="")
            retrival_pipeline(query)
        elif option == "3":
            print("Thank you :)")
            break
        else:
            print("invalid input. try again")

print("(1) to add txt file\n(2) to ask query\n(3) to exit")
start()