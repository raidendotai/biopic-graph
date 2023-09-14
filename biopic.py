#### ANALYSIS ######################################################################

from pydantic import BaseModel
from typing import List
import openai
import backoff
import json

OPENAI_MODEL = 'gpt-3.5-turbo-16k'

openai.api_key = 'YOUR_API_KEY'

demo_text = """PASTE ARTICLE TEXT HERE"""




# schema
class Character(BaseModel):
    character_name: str
    character_birthdate: str
    character_city: str
class CharacterList(BaseModel):
    characters: List[Character]
characters_schema = CharacterList.schema()

class Event(BaseModel):
    event_name: str
    event_description: str
    event_time: str
    event_place: str
class EventList(BaseModel):
    events: List[Event]
events_schema = EventList.schema()

class Place(BaseModel):
    place_name: str
    place_city: str
class PlacesList(BaseModel):
    places: List[Place]
places_schema = PlacesList.schema()


class Relation_Character_Character(BaseModel):
    character_a: str
    character_b: str
    relation_type: str
    relation_details: str
class R_C_C_List(BaseModel):
    characters_relations: List[Relation_Character_Character]    
rels_char_char_schema = R_C_C_List.schema()

class Relation_Event_Character(BaseModel):
    character: str
    event: str
    role: str
class R_E_C_List(BaseModel):
    characters_events_relations: List[Relation_Event_Character]  
rels_event_char_schema = R_E_C_List.schema()

class Relation_Place_Character(BaseModel):
    character: str
    place: str
    relation: str
class R_P_C_List(BaseModel):
    characters_places_relations: List[Relation_Place_Character]       
rels_place_char_schema = R_P_C_List.schema()





@backoff.on_exception(backoff.expo, openai.error.RateLimitError)
def gptCompletion(**kwargs):
    # exp backoff retry
    print(f'\n> gptCompletion()')
    return openai.ChatCompletion.create(**kwargs)
    
def call_gpt(context,fn_name,fn_description,schema):
    trial = 0
    while trial < 5:
        trial +=1
        try:
            print(f'-> {fn_name}')
            response= gptCompletion(
                        model= OPENAI_MODEL,
                        messages= [
                            {'role':'system','content':'You are an expert at extracting entities and relationship from a given context text'},
                            {'role':'user','content':context},
                        ],
                        functions= [
                            {
                              "name": fn_name + '_ObjectForAPI',
                              "description": fn_description,
                              "parameters": schema
                            }
                        ],
                        function_call= {"name": fn_name + '_ObjectForAPI'}
                    )
            return json.loads(response.choices[0]["message"]["function_call"]["arguments"] , strict=False)
        except Exception as e:
            print(e)
            
def build_graph(): return True




########################################

# characters
task = (characters_schema,'extractAllCharacters','extract ALL THE characters and their details from the provided context')
schema,fn_name,fn_desc = task

characters = call_gpt(
    'CONTEXT :\n```\n{CONTEXT}\n```'.replace('{CONTEXT}', demo_text.strip() ),
    fn_name,
    fn_desc,
    schema
)
print({'characters':characters})
print('\n-------------------------------------')

# events
task = (events_schema,'extractAllEvents','extract ALL THE events and their details from the provided context')
schema,fn_name,fn_desc = task
events = call_gpt(
    'CONTEXT :\n```\n{CONTEXT}\n```'.replace('{CONTEXT}', demo_text.strip() ),
    fn_name,
    fn_desc,
    schema
)
print({'events':events})
print('\n-------------------------------------')


# places
task = (places_schema,'extractAllPlaces','extract ALL THE places from the provided context')
schema,fn_name,fn_desc = task
places = call_gpt(
    'CONTEXT :\n```\n{CONTEXT}\n```'.replace('{CONTEXT}', demo_text.strip() ),
    fn_name,
    fn_desc,
    schema
)
print({'places':places})


# rels_char_char_schema
task = (rels_char_char_schema,'extractAllCharactersRelations','extract ALL THE relationships between characters from the provided context and data')
schema,fn_name,fn_desc = task
rels_char_char = call_gpt(
    'CONTEXT :\n```\n{CONTEXT}\n```CHARACTERS :\n```\n{CHARACTERS}\n```'.replace('{CONTEXT}', demo_text.strip() )
                                                                .replace('{CHARACTERS}', json.dumps(characters) ),
    fn_name,
    fn_desc,
    schema
)
print({'rels_char_char':rels_char_char})
print('\n-------------------------------------')


# rels_event_char
task = (rels_event_char_schema,'extractAllRelationsCharactersWithEvents','extract ALL THE relationships between characters and events from the provided context and data')
schema,fn_name,fn_desc = task
rels_event_char = call_gpt(
    'CONTEXT :\n```\n{CONTEXT}\n```CHARACTERS :\n```\n\n{CHARACTERS}\n```\n\n{EVENTS}\n```'
                                                                .replace('{CONTEXT}', demo_text.strip() )
                                                                .replace('{CHARACTERS}', json.dumps(characters) )
                                                                .replace('{EVENTS}', json.dumps(events) ),
    fn_name,
    fn_desc,
    schema
)
print({'rels_event_char':rels_event_char})
print('\n-------------------------------------')


# rels_place_char
task = (rels_place_char_schema,'extractAllRelationsCharactersWithPlaces','extract ALL THE relationships between characters and places from the provided context and data'
schema,fn_name,fn_desc = task
rels_place_char = call_gpt(
    'CONTEXT :\n```\n{CONTEXT}\n```CHARACTERS :\n```\n\n{CHARACTERS}\n```\n\n{PLACES}\n```'
                                                                .replace('{CONTEXT}', demo_text.strip() )
                                                                .replace('{CHARACTERS}', json.dumps(characters) )
                                                                .replace('{PLACES}', json.dumps(places) ),
    fn_name,
    fn_desc,
    schema
)
print({'rels_place_char':rels_place_char})
print('\n-------------------------------------')


with open('dump.json', "w") as file:
    json.dump(
        {
            'characters':characters,
            'events':events,
            'places':places,
            'rels_char_char':rels_char_char,
            'rels_event_char':rels_event_char,
            'rels_place_char':rels_place_char
        },
    file)





#### PLOT ######################################################################

import networkx as nx
import matplotlib.pyplot as plt
from pyvis.network import Network
import json

f = open('dump.json')
data = json.load(f)
f.close()

# Create the graph
G = nx.Graph()

print(data.keys())

for char in data['characters']['characters']:
    G.add_node(char['character_name'], **char, node_type="Character", color='black', label=char['character_name'],)
    

places = []
areas = []
for place in data['places']['places']:
    if place['place_city'] and place['place_city'] != 'Unknown':
        if place['place_city'] not in areas: areas.append({'name':place['place_city']})
    if place['place_name'] not in places:
        if place['place_city'] and place['place_city'] != 'Unknown':
            places.append({'name':place['place_name'],'area':place['place_city']})
        else:
            places.append({'name':place['place_name']})
# print(places,areas)
for area in areas: G.add_node(area['name'], **char, node_type="Area", color='navy', label=area['name'],)
for place in places:
    G.add_node(place['name'], **char, node_type="Place", color='navy', label=place['name'],)
    if place['area']:
        G.add_edge(place['name'], place['area'], **{}, color='navy' )

for event in data['events']['events']:
    # print(event)
    caption = event['event_name']
    if event['event_time'] and event['event_time'] != 'Unknown' :
        caption = f"{event['event_name']} ({event['event_time']})"
    G.add_node(event['event_name'], **char, node_type="Event", color='red', label=caption,)
    if event['event_place'] and event['event_place'] != 'Unknown':
        if event['event_place'] in [p['name'] for p in places]:
            G.add_edge(event['event_name'], event['event_place'], **{}, color='red' )

for rel in data['rels_char_char']['characters_relations']:
    if rel['character_a'] in [c['character_name'] for c in data['characters']['characters']]:
        if rel['character_b'] in [c['character_name'] for c in data['characters']['characters']]:
            G.add_edge(
                rel['character_a'],
                rel['character_b'],
                **{},
                color='black',
                label = rel['relation_type']
            )

for rel in data['rels_event_char']['characters_events_relations']:
    if rel['character'] in [c['character_name'] for c in data['characters']['characters']]:
        if rel['event'] in [c['event_name'] for c in data['events']['events']]:
        
            G.add_edge(
                rel['character'],
                rel['event'],
                **{},
                color='red',
                label = rel['role']
            )

for rel in data['rels_place_char']['characters_places_relations']:

     if rel['character'] in [c['character_name'] for c in data['characters']['characters']]:
        if rel['place'] in [p['name'] for p in places]:
            G.add_edge(
                rel['character'],
                rel['place'],
                **{},
                color='navy',
                #label = rel['relation']
            )   
    
# Create the interactive plot using pyvis
nt = Network(height="1000px", width="100%", notebook=True)
# nt.barnes_hut()
nt.from_nx(G)


nt.save_graph("interactive_graph.html")
# nt.show("interactive_graph.html")
