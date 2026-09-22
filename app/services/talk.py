#create user
#create chat
#run a while loop till -
# run chat_turn function after passing the db, chat-user and query
# talk and talk and talk and see how it performs
from app.database.crud.chat import create_chat, delete_chat
from app.database.crud.user import create_user, delete_user, delete_all_users
from app.database.model import SessionLocal
from app.services.chat_orchestration import ChatUser, handle_chat_turn
from app.services.second_brain_animation import start_second_brain

try :
    db = SessionLocal()
    name = "Rahul"
    email = "rahul@gmail.com"
    mobile_no = "1234567890"

    delete_all_users(db)
    user = create_user(db, name, email, mobile_no)
    chat = create_chat(db, user.id)

    chat_user = ChatUser(user_id = user.id, chat_id = chat.id)
    start_second_brain()

    while(True):
        query = input("Enter your query: ")
        if query == "exit":
            print("Meet you again !")
            break
        handle_chat_turn(db, chat_user, query, summarisation_threshold = 2, ingestion_threshold = 2,)

    delete_user(db, user.id)
finally:
    print("Execution completed successfully!")
