import my_agent

def ingestion_pipeline(file_path):
    documents = my_agent.load_documents(file_path)
    chunks = my_agent.split_documents(documents)
    my_agent.create_vector_store(chunks)
        
def retrival_pipeline(query):
    my_agent.ask_query(query)

def start():
    print("\n"+"-"*100)
    option = input("choise : ")
    if option == "1":
        file_path = input("directory path : ")
        ingestion_pipeline(file_path)
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

print("choose \n (1) to add txt file\n (2) to ask query\n (3) to exit\n")
start()