import uvicorn
from fastapi import FastAPI
from enum import Enum
from pydantic import BaseModel


class Category(str, Enum):
    atheism = "atheism"
    climate_change = "climate_change"
    hillary_clinton = "hillary_clinton"
    legalize_abortion = "legalize_abortion"
    feminist_movement = "feminist_movement"


class Target(BaseModel):
    model: str = 'Multiclass'
    description: str = None
    category: Category
    tweet: str


app = FastAPI(debug=True)  # decorator

"""
PATH/ROUTE OPERATION FUNCTIONS.
post: create data
put: update data
get: get data
delete: del data

coroutines: coroutines are functions whose execution you can pause (like generators)
event loop: when A happens, do B (asyncio)
async fn: define a funcion as being a coroutine --> async def name(): await stufff() like yield from stuff()
"""


@app.get("/")
async def root():
    return {'message': 'Hello there! This is an API that is able to classify tweets stances for 5 controversial '
                       'topics. That for now... but the idea is to build a tool to arbitrary classify text for over '
                       '100 different categories and able to extract best candidates insides a corpus... you will '
                       'see. For now, move to "home".'}


@app.get("/home")
async def home_page():
    return {'message': 'Home sweet home. Here you will decide what NLP task you want to solve and in which language.'}


@app.get("/home/multiclass/{category}")
async def get_model(category: Category, tweet: str = 'pio! pio!'):
    response = {'category':category, 'message': 'all rigth!'}
    if tweet:
        response.update({'tweet': tweet})
    if category == Category.atheism:
        return {'category': category, 'message': 'fast-bert for multiclass in a religious topic, all right!',
                'tweet': tweet}
    if category.value == 'climate_change':
        return {'category': category, 'message': 'fast-bert for multiclass in somthing of this nature?, of course!',
                'tweet': tweet}
    if category == Category.hillary_clinton:
        return {'category': category, 'message': 'fast-bert for multiclass about a specific politic figure, lets find '
                                                 'out!', 'tweet': tweet}
    if category == Category.legalize_abortion:
        return {'category': category, 'message': 'fast-bert for multiclass in... oh... sad... legalization of '
                                                 'abortion. Lets see.', 'tweet': tweet}
    return {'category': category, 'message': 'fast-bert for multiclass in this fast-growing movement, all right!',
            'tweet': tweet}


@app.post("/home/multiclass/")
async def get_response(query: Target):
    response = query.dict()
    # here I should run the model depending on the query params
    response.update({'prediction': 'Not implemented :S'})
    return response

if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)
