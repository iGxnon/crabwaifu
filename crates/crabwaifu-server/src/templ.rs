// Modified from crabml
// TODO: add more wacky stuff

use crabml::tensor::Tensor;
use crabml_llama2::llama2::Llama2Runner;
use crabml_llama2::model::ModelArchitecture;
use crabwaifu_common::proto::chat::{self, Message};

#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub enum ChatTemplate {
    Llama2,
    Llama3,
    Gemma,
    ChatML,
}

impl ChatTemplate {
    /// GGUF may contains a metadata called tokenizer.chat_template (maybe in a jinja format),
    /// we'd not take the chat_template directly but use a heuristic to guess the common ones.
    pub fn heuristic_guess<T: Tensor>(runner: &Llama2Runner<T>) -> Self {
        let model_name = &runner.conf().model_name;
        let model_arch = runner.conf().architecture;
        let chat_tmpl = &runner.conf().chat_template;
        if model_name.contains("gemma") || model_arch == ModelArchitecture::Gemma {
            ChatTemplate::Gemma
        } else if model_name.contains("llama2") {
            ChatTemplate::Llama2
        } else if chat_tmpl.contains("chatml") || chat_tmpl.contains("<|im_start|>") {
            ChatTemplate::ChatML
        } else if model_name.contains("llama3") || chat_tmpl.contains("<|start_header_id|>") {
            ChatTemplate::Llama3
        } else {
            // take llama2 as fallback.
            ChatTemplate::Llama2
        }
    }

    pub fn stop_mark(&self) -> &str {
        match self {
            ChatTemplate::Llama2 => "[/INST]",
            ChatTemplate::Gemma => "<end_of_turn>",
            ChatTemplate::Llama3 => "<|eot_id|>",
            ChatTemplate::ChatML => "<|im_end|>",
        }
    }

    pub fn format(&self, messages: &[Message]) -> String {
        let mut builder = PromptBuilder::new(*self);
        for msg in messages {
            builder.feed(msg);
        }
        builder.finish()
    }

    pub fn format_prompt(&self, prompt: &str) -> String {
        match self {
            ChatTemplate::Llama2 => format!("[INST] {} [/INST][[INST]]", prompt),
            ChatTemplate::Llama3 => format!(
                "<|start_header_id|>user<|end_header_id|>\n\n{}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
                prompt
            ),
            ChatTemplate::Gemma => format!("<start_of_turn>user\n{}<end_of_turn><start_of_turn>model\n", prompt),
            ChatTemplate::ChatML => format!("<|im_start|>user\n{}<|im_end|><|im_start|>assistant\n", prompt),
        }
    }
}

/// state machine
struct PromptBuilder {
    prompt: String,
    state: BuilderState,
    templ: ChatTemplate,
    register: String,
}

#[derive(Debug, Clone, Copy)]
enum BuilderState {
    BeginSystem,
    WriteSystem,
    BeginUser,
    BeginAssistant,
}

impl PromptBuilder {
    fn new(templ: ChatTemplate) -> Self {
        Self {
            prompt: String::new(),
            state: BuilderState::BeginSystem,
            templ,
            register: String::new(),
        }
    }

    fn feed(&mut self, message: &Message) {
        match (message.role, self.state) {
            (chat::Role::System, BuilderState::BeginSystem) => match self.templ {
                ChatTemplate::Llama2 => {
                    self.prompt.push_str("[INST] ");
                    self.register.clone_from(&message.content);
                    self.state = BuilderState::WriteSystem;
                }
                ChatTemplate::Llama3 => {
                    self.prompt.push_str(&format!(
                        "<|start_header_id|>system<|end_header_id|>\n\n{}<|eot_id|>",
                        message.content
                    ));
                    self.state = BuilderState::BeginUser;
                }
                ChatTemplate::Gemma => {
                    self.prompt.push_str("<start_of_turn>user\n");
                    self.register.clone_from(&message.content);
                    self.state = BuilderState::WriteSystem;
                }
                ChatTemplate::ChatML => {
                    self.prompt.push_str(&format!(
                        "<|im_start|>system\n{}<|im_end|>",
                        message.content
                    ));
                    self.state = BuilderState::BeginUser;
                }
            },
            (chat::Role::User, BuilderState::BeginSystem) => match self.templ {
                ChatTemplate::Llama2 => {
                    self.prompt
                        .push_str(&format!("[INST] {} [/INST]", message.content));
                    self.state = BuilderState::BeginAssistant;
                }
                ChatTemplate::Llama3 => {
                    self.prompt.push_str(&format!(
                        "<|start_header_id|>user<|end_header_id|>\n\n{}<|eot_id|>",
                        message.content
                    ));
                    self.state = BuilderState::BeginAssistant;
                }
                ChatTemplate::Gemma => {
                    self.prompt.push_str(&format!(
                        "<start_of_turn>user\n{}<end_of_turn>",
                        message.content
                    ));
                    self.state = BuilderState::BeginAssistant
                }
                ChatTemplate::ChatML => {
                    self.prompt
                        .push_str(&format!("<|im_start|>user\n{}<|im_end|>", message.content));
                    self.state = BuilderState::BeginAssistant;
                }
            },
            (chat::Role::User, BuilderState::WriteSystem) => match self.templ {
                ChatTemplate::Llama2 => {
                    self.prompt.push_str(&format!(
                        "<<SYS>>{}<</SYS>> {} [/INST]",
                        self.register, message.content
                    ));
                    self.register.clear();
                    self.state = BuilderState::BeginAssistant;
                }
                ChatTemplate::Llama3 => {}
                ChatTemplate::Gemma => {
                    self.prompt.push_str(&format!(
                        "{} {}<end_of_turn>",
                        self.register, message.content,
                    ));
                    self.register.clear();
                    self.state = BuilderState::BeginAssistant;
                }
                ChatTemplate::ChatML => {}
            },
            (chat::Role::Assistant, BuilderState::BeginAssistant) => match self.templ {
                ChatTemplate::Llama2 => {
                    self.prompt
                        .push_str(&format!("[[INST]] {} [/INST]", message.content));
                    self.state = BuilderState::BeginUser;
                }
                ChatTemplate::Llama3 => {
                    self.prompt.push_str(&format!(
                        "<|start_header_id|>assistant<|end_header_id|>\n\n{}<|eot_id|>",
                        message.content
                    ));
                    self.state = BuilderState::BeginUser;
                }
                ChatTemplate::Gemma => {
                    self.prompt.push_str(&format!(
                        "<start_of_turn>model\n{}<end_of_turn>",
                        message.content
                    ));
                    self.state = BuilderState::BeginUser;
                }
                ChatTemplate::ChatML => {
                    self.prompt.push_str(&format!(
                        "<|im_start|>assistant\n{}<|im_end|>",
                        message.content
                    ));
                    self.state = BuilderState::BeginUser;
                }
            },
            (chat::Role::User, BuilderState::BeginUser) => match self.templ {
                ChatTemplate::Llama2 => {
                    self.prompt
                        .push_str(&format!("[INST] {} [/INST]", message.content));
                    self.state = BuilderState::BeginAssistant;
                }
                ChatTemplate::Llama3 => {
                    self.prompt.push_str(&format!(
                        "<|start_header_id|>user<|end_header_id|>\n\n{}<|eot_id|>",
                        message.content
                    ));
                    self.state = BuilderState::BeginAssistant;
                }
                ChatTemplate::Gemma => {
                    self.prompt.push_str(&format!(
                        "<start_of_turn>user\n{}<end_of_turn>",
                        message.content
                    ));
                    self.state = BuilderState::BeginAssistant
                }
                ChatTemplate::ChatML => {
                    self.prompt
                        .push_str(&format!("<|im_start|>user\n{}<|im_end|>", message.content));
                    self.state = BuilderState::BeginAssistant;
                }
            },
            (role, state) => {
                log::warn!("ignore message role {role:?} from state {state:?}");
            }
        }
    }

    fn finish(mut self) -> String {
        if matches!(self.state, BuilderState::BeginAssistant) {
            match self.templ {
                ChatTemplate::Llama2 => {
                    self.prompt.push_str("[[INST]]");
                }
                ChatTemplate::Llama3 => {
                    self.prompt
                        .push_str("<|start_header_id|>assistant<|end_header_id|>\n\n");
                }
                ChatTemplate::Gemma => {
                    self.prompt.push_str("<start_of_turn>model\n");
                }
                ChatTemplate::ChatML => {
                    self.prompt.push_str("<|im_start|>assistant\n");
                }
            }
        }
        self.prompt
    }
}

pub struct ChatReplyIterator<'a, T> {
    inner: T,
    stop_marks: Vec<String>,
    stop_mark_matcher: MarkMatcher,
    has_stop_mark: &'a mut bool,
}

impl<'a, T> ChatReplyIterator<'a, T>
where
    T: Iterator<Item = anyhow::Result<String>>,
{
    pub fn new(inner: T, stop_marks: Vec<String>, has_stop_mark: &'a mut bool) -> Self {
        Self {
            inner,
            stop_marks: stop_marks.clone(),
            stop_mark_matcher: MarkMatcher::new(stop_marks),
            has_stop_mark,
        }
    }
}

impl<'a, T> Iterator for ChatReplyIterator<'a, T>
where
    T: Iterator<Item = anyhow::Result<String>>,
{
    type Item = anyhow::Result<String>;

    fn next(&mut self) -> Option<Self::Item> {
        if *self.has_stop_mark {
            return None;
        }

        loop {
            let token = match self.inner.next() {
                None => return None,
                Some(Err(err)) => return Some(Err(err)),
                Some(Ok(token)) => token,
            };

            let token = match self.stop_mark_matcher.push(token) {
                None => continue,
                Some(token) => token,
            };

            if self.stop_marks.contains(&token) {
                *self.has_stop_mark = true;
                return None;
            }

            return Some(Ok(token));
        }
    }
}

pub struct MarkMatcher {
    state: MarkMatchState,
    buf: String,
    marks: Vec<String>,
}

pub enum MarkMatchState {
    Inactive,
    Active,
}

impl MarkMatcher {
    pub fn new(marks: Vec<String>) -> Self {
        Self {
            state: MarkMatchState::Inactive,
            buf: String::new(),
            marks,
        }
    }

    pub fn push(&mut self, token: String) -> Option<String> {
        match self.state {
            MarkMatchState::Inactive => {
                // exact match, do not change state
                if self.marks.contains(&token) {
                    return Some(token);
                }

                // got any partial match, change state to active, and push the token
                // to the buffer, and wait for the rest of the mark.
                if self.marks.iter().any(|m| m.starts_with(&token)) {
                    self.state = MarkMatchState::Active;
                    self.buf = token;
                    return None;
                }

                // no match, return the token directly
                Some(token)
            }
            MarkMatchState::Active => {
                self.buf.push_str(&token);

                // exact match, change state to inactive, and return the buffer
                if self.marks.contains(&self.buf) {
                    self.state = MarkMatchState::Inactive;
                    return Some(self.buf.clone());
                }

                // not match anymore, return the buffer directly
                if !self.marks.iter().any(|m| m.starts_with(&self.buf)) {
                    self.state = MarkMatchState::Inactive;
                    return Some(self.buf.clone());
                }

                // partial match, wait for the rest of the mark
                None
            }
        }
    }
}

#[cfg(test)]
mod test {
    use crabwaifu_common::proto::chat::{Message, Role};

    use super::{ChatReplyIterator, PromptBuilder};

    #[test]
    fn test_llama2_builder_works() {
        {
            let mut builder = PromptBuilder::new(super::ChatTemplate::Llama2);
            builder.feed(&Message {
                role: Role::System,
                content: "I am the system message".to_string(),
            });
            builder.feed(&Message {
                role: Role::User,
                content: "I ams the user message 1".to_string(),
            });
            builder.feed(&Message {
                role: Role::Assistant,
                content: "I ams the assistant message 1".to_string(),
            });
            builder.feed(&Message {
                role: Role::User,
                content: "I ams the user message 2".to_string(),
            });

            builder.feed(&Message {
                role: Role::User,
                content: "BAD!!!".to_string(),
            });

            builder.feed(&Message {
                role: Role::Assistant,
                content: "I ams the assistant message 2".to_string(),
            });
            builder.feed(&Message {
                role: Role::User,
                content: "I ams the user message 3".to_string(),
            });
            let prompt = builder.finish();
            assert_eq!(prompt, "[INST] <<SYS>>I am the system message<</SYS>> I ams the user message 1 [/INST][[INST]] I ams the assistant message 1 [/INST][INST] I ams the user message 2 [/INST][[INST]] I ams the assistant message 2 [/INST][INST] I ams the user message 3 [/INST][[INST]]");
        }

        {
            let mut builder = PromptBuilder::new(super::ChatTemplate::Llama2);
            builder.feed(&Message {
                role: Role::User,
                content: "I ams the user message 1".to_string(),
            });
            builder.feed(&Message {
                role: Role::Assistant,
                content: "I ams the assistant message 1".to_string(),
            });
            builder.feed(&Message {
                role: Role::User,
                content: "I ams the user message 2".to_string(),
            });

            builder.feed(&Message {
                role: Role::User,
                content: "BAD!!!".to_string(),
            });

            builder.feed(&Message {
                role: Role::Assistant,
                content: "I ams the assistant message 2".to_string(),
            });
            builder.feed(&Message {
                role: Role::User,
                content: "I ams the user message 3".to_string(),
            });

            let prompt = builder.finish();
            assert_eq!(prompt, "[INST] I ams the user message 1 [/INST][[INST]] I ams the assistant message 1 [/INST][INST] I ams the user message 2 [/INST][[INST]] I ams the assistant message 2 [/INST][INST] I ams the user message 3 [/INST][[INST]]");
        }
    }

    #[test]
    fn test_llama3_builder_works() {
        {
            let mut builder = PromptBuilder::new(super::ChatTemplate::Llama3);
            builder.feed(&Message {
                role: Role::System,
                content: "I am the system message".to_string(),
            });
            builder.feed(&Message {
                role: Role::User,
                content: "I ams the user message 1".to_string(),
            });
            builder.feed(&Message {
                role: Role::Assistant,
                content: "I ams the assistant message 1".to_string(),
            });
            builder.feed(&Message {
                role: Role::User,
                content: "I ams the user message 2".to_string(),
            });

            builder.feed(&Message {
                role: Role::User,
                content: "BAD!!!".to_string(),
            });

            builder.feed(&Message {
                role: Role::Assistant,
                content: "I ams the assistant message 2".to_string(),
            });
            builder.feed(&Message {
                role: Role::User,
                content: "I ams the user message 3".to_string(),
            });
            let prompt = builder.finish();
            assert_eq!(prompt, "<|start_header_id|>system<|end_header_id|>\n\nI am the system message<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nI ams the user message 1<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\nI ams the assistant message 1<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nI ams the user message 2<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\nI ams the assistant message 2<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nI ams the user message 3<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n");
        }

        {
            let mut builder = PromptBuilder::new(super::ChatTemplate::Llama3);
            builder.feed(&Message {
                role: Role::User,
                content: "I ams the user message 1".to_string(),
            });
            builder.feed(&Message {
                role: Role::Assistant,
                content: "I ams the assistant message 1".to_string(),
            });
            builder.feed(&Message {
                role: Role::User,
                content: "I ams the user message 2".to_string(),
            });

            builder.feed(&Message {
                role: Role::User,
                content: "BAD!!!".to_string(),
            });

            builder.feed(&Message {
                role: Role::Assistant,
                content: "I ams the assistant message 2".to_string(),
            });
            builder.feed(&Message {
                role: Role::User,
                content: "I ams the user message 3".to_string(),
            });

            let prompt = builder.finish();
            assert_eq!(prompt, "<|start_header_id|>user<|end_header_id|>\n\nI ams the user message 1<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\nI ams the assistant message 1<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nI ams the user message 2<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\nI ams the assistant message 2<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nI ams the user message 3<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n");
        }
    }

    #[test]
    fn test_gemma_builder_works() {
        {
            let mut builder = PromptBuilder::new(super::ChatTemplate::Gemma);
            builder.feed(&Message {
                role: Role::System,
                content: "I am the system message".to_string(),
            });
            builder.feed(&Message {
                role: Role::User,
                content: "I ams the user message 1".to_string(),
            });
            builder.feed(&Message {
                role: Role::Assistant,
                content: "I ams the assistant message 1".to_string(),
            });
            builder.feed(&Message {
                role: Role::User,
                content: "I ams the user message 2".to_string(),
            });

            builder.feed(&Message {
                role: Role::User,
                content: "BAD!!!".to_string(),
            });

            builder.feed(&Message {
                role: Role::Assistant,
                content: "I ams the assistant message 2".to_string(),
            });
            builder.feed(&Message {
                role: Role::User,
                content: "I ams the user message 3".to_string(),
            });
            let prompt = builder.finish();
            assert_eq!(prompt, "<start_of_turn>user\nI am the system message I ams the user message 1<end_of_turn><start_of_turn>model\nI ams the assistant message 1<end_of_turn><start_of_turn>user\nI ams the user message 2<end_of_turn><start_of_turn>model\nI ams the assistant message 2<end_of_turn><start_of_turn>user\nI ams the user message 3<end_of_turn><start_of_turn>model\n");
        }

        {
            let mut builder = PromptBuilder::new(super::ChatTemplate::Gemma);
            builder.feed(&Message {
                role: Role::User,
                content: "I ams the user message 1".to_string(),
            });
            builder.feed(&Message {
                role: Role::Assistant,
                content: "I ams the assistant message 1".to_string(),
            });
            builder.feed(&Message {
                role: Role::User,
                content: "I ams the user message 2".to_string(),
            });

            builder.feed(&Message {
                role: Role::User,
                content: "BAD!!!".to_string(),
            });

            builder.feed(&Message {
                role: Role::Assistant,
                content: "I ams the assistant message 2".to_string(),
            });
            builder.feed(&Message {
                role: Role::User,
                content: "I ams the user message 3".to_string(),
            });

            let prompt = builder.finish();
            assert_eq!(prompt, "<start_of_turn>user\nI ams the user message 1<end_of_turn><start_of_turn>model\nI ams the assistant message 1<end_of_turn><start_of_turn>user\nI ams the user message 2<end_of_turn><start_of_turn>model\nI ams the assistant message 2<end_of_turn><start_of_turn>user\nI ams the user message 3<end_of_turn><start_of_turn>model\n");
        }
    }

    #[test]
    fn test_chatml_builder_works() {
        {
            let mut builder = PromptBuilder::new(super::ChatTemplate::ChatML);
            builder.feed(&Message {
                role: Role::System,
                content: "I am the system message".to_string(),
            });
            builder.feed(&Message {
                role: Role::User,
                content: "I ams the user message 1".to_string(),
            });
            builder.feed(&Message {
                role: Role::Assistant,
                content: "I ams the assistant message 1".to_string(),
            });
            builder.feed(&Message {
                role: Role::User,
                content: "I ams the user message 2".to_string(),
            });

            builder.feed(&Message {
                role: Role::User,
                content: "BAD!!!".to_string(),
            });

            builder.feed(&Message {
                role: Role::Assistant,
                content: "I ams the assistant message 2".to_string(),
            });
            builder.feed(&Message {
                role: Role::User,
                content: "I ams the user message 3".to_string(),
            });
            let prompt = builder.finish();
            assert_eq!(prompt, "<|im_start|>system\nI am the system message<|im_end|><|im_start|>user\nI ams the user message 1<|im_end|><|im_start|>assistant\nI ams the assistant message 1<|im_end|><|im_start|>user\nI ams the user message 2<|im_end|><|im_start|>assistant\nI ams the assistant message 2<|im_end|><|im_start|>user\nI ams the user message 3<|im_end|><|im_start|>assistant\n");
        }

        {
            let mut builder = PromptBuilder::new(super::ChatTemplate::ChatML);
            builder.feed(&Message {
                role: Role::User,
                content: "I ams the user message 1".to_string(),
            });
            builder.feed(&Message {
                role: Role::Assistant,
                content: "I ams the assistant message 1".to_string(),
            });
            builder.feed(&Message {
                role: Role::User,
                content: "I ams the user message 2".to_string(),
            });

            builder.feed(&Message {
                role: Role::User,
                content: "BAD!!!".to_string(),
            });

            builder.feed(&Message {
                role: Role::Assistant,
                content: "I ams the assistant message 2".to_string(),
            });
            builder.feed(&Message {
                role: Role::User,
                content: "I ams the user message 3".to_string(),
            });

            let prompt = builder.finish();
            assert_eq!(prompt, "<|im_start|>user\nI ams the user message 1<|im_end|><|im_start|>assistant\nI ams the assistant message 1<|im_end|><|im_start|>user\nI ams the user message 2<|im_end|><|im_start|>assistant\nI ams the assistant message 2<|im_end|><|im_start|>user\nI ams the user message 3<|im_end|><|im_start|>assistant\n");
        }
    }

    #[test]
    fn test_chat_reply_iter() {
        let iter = [
            "I", "love", "u", "<|end_of", "_turn|>", "<e", "os>", "I", "hate", "you",
        ]
        .iter()
        .map(ToString::to_string)
        .map(Ok);

        let mut has_stop_mark = false;
        let mut chat_iter = ChatReplyIterator::new(
            iter,
            vec!["<eos>".to_string(), "<|end_of_turn|>".to_string()],
            &mut has_stop_mark,
        );

        assert_eq!(chat_iter.next().unwrap().unwrap(), "I");
        assert_eq!(chat_iter.next().unwrap().unwrap(), "love");
        assert_eq!(chat_iter.next().unwrap().unwrap(), "u");
        assert!(chat_iter.next().is_none());
        assert!(chat_iter.next().is_none());
        assert!(has_stop_mark);

        let iter = ["I", "love", "u"].iter().map(ToString::to_string).map(Ok);

        has_stop_mark = false;
        let mut chat_iter = ChatReplyIterator::new(
            iter,
            vec!["<eos>".to_string(), "<|end_of_turn|>".to_string()],
            &mut has_stop_mark,
        );

        assert_eq!(chat_iter.next().unwrap().unwrap(), "I");
        assert_eq!(chat_iter.next().unwrap().unwrap(), "love");
        assert_eq!(chat_iter.next().unwrap().unwrap(), "u");
        assert!(chat_iter.next().is_none());
        assert!(chat_iter.next().is_none());
        assert!(!has_stop_mark);
    }
}
