-- Tables
CREATE TABLE IF NOT EXISTS users (
    id SERIAL PRIMARY KEY,
    username VARCHAR(50) NOT NULL,
    pass_hash VARCHAR(1024) NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_users_username ON users (username);
-- Unique index for username
ALTER TABLE users ADD CONSTRAINT unique_username UNIQUE (username);

CREATE TABLE IF NOT EXISTS models (
    id SERIAL PRIMARY KEY,
    name VARCHAR(50) NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_models_name ON models (name);
-- Unique index for name
ALTER TABLE models ADD CONSTRAINT unique_name UNIQUE (name);

CREATE TABLE IF NOT EXISTS history (
    id SERIAL PRIMARY KEY,
    user_id INT NOT NULL,
    model_id INT NOT NULL,
    context TEXT NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE
    FOREIGN KEY (model_id) REFERENCES models (id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_history_user_id ON history (user_id);
ALTER TABLE history ADD CONSTRAINT unique_history_user_id UNIQUE (user_id);

-- Trigger to auto update updated_at column
CREATE
OR REPLACE FUNCTION manage_updated_at(_tbl regclass) RETURNS VOID AS $$ 
BEGIN EXECUTE format(
    'CREATE TRIGGER set_updated_at BEFORE UPDATE ON %s FOR EACH ROW EXECUTE PROCEDURE set_updated_at()',
    _tbl
);
END;
$$ LANGUAGE plpgsql;

-- Function to auto update updated_at column
CREATE
OR REPLACE FUNCTION set_updated_at() RETURNS trigger AS $$ 
BEGIN IF (
        NEW IS DISTINCT
        FROM OLD
            AND NEW.updated_at IS NOT DISTINCT
                FROM OLD.updated_at
    ) THEN NEW.updated_at := NOW();
END IF;
RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Auto update updated_at column on all tables that have it
CREATE
OR REPLACE FUNCTION create_trigger_if_updated_at_exists() RETURNS VOID AS $$
DECLARE table_record RECORD;
BEGIN FOR table_record IN
    SELECT table_name
    FROM information_schema.columns
    WHERE column_name = 'updated_at' 
LOOP PERFORM manage_updated_at(table_record.table_name::regclass);
END LOOP;
END;
$$ LANGUAGE plpgsql;

-- Run the function to auto update updated_at column
SELECT
    create_trigger_if_updated_at_exists();
