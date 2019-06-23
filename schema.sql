-- The schema for a PostgreSQL database storing cached models.

DROP TABLE IF EXISTS models CASCADE;
CREATE TABLE models (
	id              SERIAL CONSTRAINT model_id PRIMARY KEY,
	location        VARCHAR(100) NOT NULL UNIQUE,
	name            VARCHAR( 80) NOT NULL,
	artist          VARCHAR( 80),
	album           VARCHAR( 80),
	year            SMALLINT,
	bpm             SMALLINT,
	time_sig_top    SMALLINT,
	time_sig_bottom SMALLINT
);

-- Tags encompass genres, instruments, "feels", etc
DROP TABLE IF EXISTS tags CASCADE;
CREATE TABLE tags (
	id   SERIAL CONSTRAINT tag_id PRIMARY KEY,
	name varchar(50) NOT NULL UNIQUE
);

DROP TABLE IF EXISTS tagmap;
CREATE TABLE tagmap (
	model_id INTEGER NOT NULL,
	tag_id   INTEGER NOT NULL,
	CONSTRAINT fk_model_id FOREIGN KEY (model_id) REFERENCES models(id) ON DELETE CASCADE ON UPDATE CASCADE,
	CONSTRAINT fk_tag_id FOREIGN KEY (tag_id) REFERENCES tags(id) ON DELETE CASCADE ON UPDATE CASCADE
);

GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO midgenuser;
GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO midgenuser;
