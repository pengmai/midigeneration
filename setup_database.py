"""Script to populate the database with albums"""

import records
import yaml

from model import Model

def insert_albums(album_name='albums.yaml'):
    with open(album_name, 'r') as f:
        albums = yaml.safe_load(f)

    # Expects a DATABASE_URL environment variable to be set.
    db = records.Database()
    for album in albums:
        for song in album['songs']:
            songname = song['name']
            print(f'Training model from {songname}')
            m = Model.from_album(song, album, train=True, pickle=True)
            m.insert_into_db(db)
    print('Complete.')

def get_model_from_query(query):
    db = records.Database()
    rows = db.query(query).all()
    if not rows:
        raise Exception(f'Query returned no models: {query}')
    model = Model.load_from_disk(rows[0].location)
    for row in rows[1:]:
        model.add_model(Model.load_from_disk(row.location))
    return model

if __name__ == '__main__':
    insert_albums()
