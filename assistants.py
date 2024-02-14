import os
import time
import logging


def generate_response(client,message_body):
    thread = client.beta.threads.create()

    #Add a message to the thread
    message = client.beta.threads.messages.create(
        thread_id=thread.id,
        role = 'user',
        content = message_body,
    )

    reponse = run_assistant(client,thread)

    return reponse

def run_assistant(client,thread):

    assistant = client.beta.assistants.retrieve("asst_TxN0uooF3BvQgcrhoVUf3jUb")
    # thread = client.beta.threads.retrieve(thread_id)

    run = client.beta.threads.runs.create(
        thread_id=thread.id,
        assistant_id=assistant.id
    )

    while run.status != "completed":
        time.sleep(0.9)
        run = client.beta.threads.runs.retrieve(thread_id=thread.id, run_id=run.id)

    messages = client.beta.threads.messages.list(thread_id = thread.id)

    new_message = messages.data[0].content[0].text.value

    logging.info(f"Generated message: {new_message}")
    return new_message