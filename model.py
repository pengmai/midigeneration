import hashlib
import os
import pickle
import midi_io

from data import Piece
from models import Markov, HiddenMarkov, NoteState

class Model:
    """
    A representation of a model for a single song as stored in the database.
    """
    def __init__(self, name, location, train=False, pickle=False):
        self.name = name
        self.midi_file = location
        self.artist = None
        self.album = None
        self.year = None
        self.tags = []
        self.location = '.cached/model-' + hashlib.sha256(self.midi_file.encode()).hexdigest()

        self.hmm = HiddenMarkov()
        self.mm = Markov()
        self.piece = Piece(self.midi_file)
        self.time_sig_top = self.piece.tracks[0].time_top
        self.time_sig_bottom = self.piece.tracks[0].time_bottom
        self.bpm = self.piece.tracks[0].bpm

        if os.path.exists(self.location):
            print('Cached model found, skipping training/serialization')
            return
        if train:
            self.train()
        if pickle:
            self.save_to_disk()

    @staticmethod
    def from_album(song, album, train=False, pickle=False):
        m = Model(song['name'], song['location'], train=train, pickle=pickle)
        m.artist = album['artist'] if 'artist' in album else None
        m.album = album['name'] if 'name' in album else None
        m.year = album['year'] if 'year' in album else None
        m.tags = get_tags(album, song)
        return m

    def train(self):
        key_sig, state_chain, all_bars, obs = NoteState.piece_to_state_chain(self.piece)
        self.hmm.add(all_bars)
        self.hmm.train_obs(obs, key_sig)
        self.mm.add(state_chain)

    def generate(self, output='output/output.mid', save=True):
        """Generates a song that can be serialized via the midi library."""
        hidden_chain = self.hmm.generate_hidden()
        state_chain = self.hmm.generate(hidden_chain, self.mm.markov)
        notes = NoteState.state_chain_to_notes(state_chain, self.piece.bar)
        song = [self.piece.meta] + [[ n.note_event() for n in notes ]]
        if save:
            midi_io.write(output, song)
        return hidden_chain, state_chain

    def add_model(self, other):
        """
        Adds the internal markov models to this one.
        Parameters
        ----------
        other : Model
            The other model to add.
        """
        self.hmm = self.hmm.add_model(other.hmm)
        self.mm = self.mm.add_model(other.mm)

    def save_to_disk(self):
        with open(self.location, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load_from_disk(location: str):
        with open(location, 'rb') as f:
            model = pickle.load(f)
        return model

    def insert_into_db(self, db):
        """
        Parameters
        ----------
        db : records.Database
            An open database connection.
        """
        row = db.query('SELECT * FROM models WHERE location = :l', l=self.location).first()
        if row:
            print('Model found in database, skipping')
            return

        res = db.query('''
            INSERT INTO models (
                location,
                name,
                artist,
                album,
                year,
                bpm,
                time_sig_top,
                time_sig_bottom
            ) VALUES (
                :location,
                :name,
                :artist,
                :album,
                :year,
                :bpm,
                :time_sig_top,
                :time_sig_bottom
            ) RETURNING id''',
            location=self.location, name=self.name, artist=self.artist,
            album=self.album, year=self.year, bpm=self.bpm,
            time_sig_top=self.time_sig_top, time_sig_bottom=self.time_sig_bottom)
        model_id = res.first().id
        for tag_id in self.get_or_insert_tag_ids(db):
            # Link the tags to the model.
            db.query('INSERT INTO tagmap (model_id, tag_id) VALUES (:m, :t)',
                     m=model_id, t=tag_id)

    def get_or_insert_tag_ids(self, db):
        def get_or_insert_tag(tag):
            row = db.query('SELECT id FROM tags WHERE name = :tag', tag=tag).first()
            if row:
                return row.id
            else:
                res = db.query('INSERT INTO tags (name) VALUES (:tag) RETURNING id', tag=tag)
                return res.first().id
        return [get_or_insert_tag(tag) for tag in self.tags]

def get_tags(album, song):
    album_tags = album['tags'] if 'tags' in album else []
    song_tags = song['tags'] if 'tags' in song else []
    return album_tags + song_tags
