use std::env;
use std::sync::OnceLock;

use pbkdf2::password_hash::rand_core::OsRng;
use pbkdf2::password_hash::{PasswordHash, PasswordHasher, PasswordVerifier, SaltString};
use pbkdf2::{Params, Pbkdf2};
use sqlx::postgres::PgPoolOptions;

static POOL: OnceLock<sqlx::Pool<sqlx::Postgres>> = OnceLock::new();

pub fn pool() -> &'static sqlx::Pool<sqlx::Postgres> {
    POOL.get_or_init(init_pool)
}

fn init_pool() -> sqlx::Pool<sqlx::Postgres> {
    PgPoolOptions::new()
        .max_connections(100)
        .connect_lazy(
            &env::var("DATABASE_URL").unwrap_or(
                "postgres://yui:yui@127.0.0.1:5432/crabwaifu?sslmode=disable".to_string(),
            ),
        )
        .expect("db connect failed")
}

static PBKDF_2: OnceLock<PBKDF2> = OnceLock::new();

pub fn pbkdf2() -> &'static PBKDF2 {
    PBKDF_2.get_or_init(PBKDF2::new)
}

pub struct PBKDF2 {
    salt: SaltString,
}

impl PBKDF2 {
    fn new() -> Self {
        Self {
            salt: SaltString::generate(&mut OsRng),
        }
    }

    pub fn key(&self, pwd: String) -> String {
        Pbkdf2
            .hash_password_customized(
                pwd.as_bytes(),
                None,
                None,
                Params {
                    rounds: 4096,
                    output_length: 32,
                },
                &self.salt,
            )
            .unwrap()
            .to_string()
    }

    pub fn verify(&self, password: String, phc: String) -> bool {
        let parsed_hash = PasswordHash::new(&phc);
        let Ok(hash) = parsed_hash else {
            return false;
        };
        Pbkdf2.verify_password(password.as_bytes(), &hash).is_ok()
    }
}

#[cfg(test)]
mod test {
    use super::*;
    #[test]
    fn test_hash() {
        let phc = pbkdf2().key("pwd1".to_string());
        let check = pbkdf2().verify("pwd1".to_string(), phc);
        assert!(check);
    }
}
