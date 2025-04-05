use std::rc::Rc;
use std::sync::Mutex;

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use crabwaifu_common::network::{Rx, Tx};
use futures::Stream;
use gtk::prelude::*;
use gtk::{
    glib, Application, ApplicationWindow, Box as GtkBox, Button, Entry, Image, PasswordEntry,
    ScrolledWindow, TextView,
};

use crate::client::Client;

const APP_ID: &str = "com.example.chat";

pub async fn run_ui(client: &mut Client<impl Tx, impl Rx>) -> anyhow::Result<()> {
    // Create a new application
    let app = Application::builder().application_id(APP_ID).build();

    // Connect to "activate" signal of `app`
    app.connect_activate(build_login_ui(client));

    // Run the application
    app.run_with_args(&Vec::<String>::new());

    Ok(())
}

fn build_ui(client: *mut Client<impl Tx, impl Rx>) -> impl Fn(&Application) + 'static {
    move |app: &Application| {
        // Create a new window
        let window = ApplicationWindow::builder()
            .application(app)
            .title("Chat App")
            .default_width(600)
            .default_height(800)
            .build();

        // Create a vertical box to hold our widgets
        let main_box = GtkBox::builder()
            .orientation(gtk::Orientation::Vertical)
            .spacing(6)
            .margin_top(6)
            .margin_bottom(6)
            .margin_start(6)
            .margin_end(6)
            .build();

        // Create a ScrolledWindow to contain the TextView
        let scrolled_window = ScrolledWindow::builder()
            .hexpand(true)
            .vexpand(true)
            .build();

        // Create TextView for displaying messages
        let text_view = TextView::builder()
            .editable(false)
            .wrap_mode(gtk::WrapMode::Word)
            .build();

        let css_provider = gtk::CssProvider::new();
        css_provider.load_from_data("textview { font-size: 14pt; }");
        text_view
            .style_context()
            .add_provider(&css_provider, gtk::STYLE_PROVIDER_PRIORITY_APPLICATION);

        scrolled_window.set_child(Some(&text_view));

        // Create a horizontal box for input area
        let input_box = GtkBox::builder()
            .orientation(gtk::Orientation::Horizontal)
            .spacing(6)
            .build();

        // Create Entry for message input
        let entry = Entry::builder()
            .hexpand(true)
            .placeholder_text("Type your message...")
            .build();

        // Create Send button
        let send_button = Button::builder().label("Send").build();

        // Create a Record button with an icon and circular style
        let record_button = Button::builder().build();
        let record_icon = Image::from_icon_name("media-record-symbolic"); // Use a record icon
        record_button.set_child(Some(&record_icon));
        record_button.style_context().add_class("circular"); // Add circular style class

        // Add widgets to input_box
        input_box.append(&entry);
        input_box.append(&send_button);
        input_box.append(&record_button);

        // Add all widgets to main_box
        main_box.append(&scrolled_window);
        main_box.append(&input_box);

        // Set main_box as the window's child
        window.set_child(Some(&main_box));

        // Get the buffer from the TextView
        let buffer = text_view.buffer();

        // Connect the "clicked" signal of the send button
        send_button.connect_clicked({
            let entry = entry.clone();
            let buffer = buffer.clone();
            move |_| {
                let text = entry.text().to_string();
                if !text.is_empty() {
                    // Format the message
                    let tag = buffer.create_tag(None, &[("foreground", &"blue")]).unwrap();
                    buffer.insert_with_tags(&mut buffer.end_iter(), "You: ", &[&tag]);
                    buffer.insert(&mut buffer.end_iter(), &format!("{}\n", text));

                    let client = unsafe { &mut *client };

                    let ctx = glib::MainContext::default();
                    ctx.spawn_local({
                        let buffer = buffer.clone();
                        let text_view = text_view.clone();
                        async move {
                            let reply = client.stream(text).await.unwrap();
                            response_text(buffer, reply, text_view).await;
                        }
                    });

                    entry.set_text("");
                }
            }
        });

        // Connect the "activate" signal of the entry (when Enter is pressed)
        entry.connect_activate({
            let send_button = send_button.clone();
            move |_| {
                send_button.emit_clicked();
            }
        });

        // Create a state to track recording status
        let is_pausing = Rc::new(std::cell::RefCell::new(true));
        // Create a shared state for the audio stream
        let audio_stream = Rc::new(Mutex::new(None));

        // Connect the Record button to handle recording logic
        record_button.connect_clicked({
            let is_pausing = is_pausing.clone();
            let record_icon = record_icon.clone();
            let audio_stream = audio_stream.clone();
            move |_| {
                let mut pausing = is_pausing.borrow_mut();
                if *pausing {
                    // Start recording
                    let host = cpal::default_host();
                    let device = host
                        .default_input_device()
                        .expect("No input device available");
                    let config = device.default_input_config().unwrap();

                    let client = unsafe { &mut *client };
                    let stream = match config.sample_format() {
                        cpal::SampleFormat::F32 => device.build_input_stream(
                            &config.into(),
                            move |data: &[f32], _| {
                                // TODO: transfer data to server
                            },
                            |err| {
                                eprintln!("Error during recording: {:?}", err);
                            },
                            None,
                        ),
                        _ => panic!("Unsupported sample format"),
                    }
                    .unwrap();

                    stream.play().unwrap();
                    *audio_stream.lock().unwrap() = Some(stream);
                    println!("Recording started");
                    record_icon.set_icon_name(Some("media-playback-stop-symbolic"));
                } else {
                    // Stop recording

                    let client = unsafe { &mut *client };
                    if let Some(stream) = audio_stream.lock().unwrap().take() {
                        stream.pause().unwrap();
                        // TODO: send stop record to server, wait response
                    }
                    println!("Recording stopped");
                    record_icon.set_icon_name(Some("media-record-symbolic"));
                }
                *pausing = !*pausing;
            }
        });

        // Show the window
        window.present();
    }
}

pub fn build_login_ui(client: *mut Client<impl Tx, impl Rx>) -> impl Fn(&Application) + 'static {
    move |app: &Application| {
        // Create a new window for login
        let login_window = ApplicationWindow::builder()
            .application(app)
            .title("Login")
            .default_width(500) // Increased width
            .default_height(400) // Increased height
            .build();

        // Create a vertical box to hold widgets
        let login_box = GtkBox::builder()
            .orientation(gtk::Orientation::Vertical)
            .spacing(20) // Increased spacing for better layout
            .margin_top(30) // Increased margins
            .margin_bottom(30)
            .margin_start(30)
            .margin_end(30)
            .build();

        // Create a horizontal box for username
        let username_row = GtkBox::builder()
            .orientation(gtk::Orientation::Horizontal)
            .spacing(10)
            .build();
        let username_icon = Image::from_icon_name("avatar-default-symbolic"); // Updated icon
        let username_entry = Entry::builder()
            .placeholder_text("Enter your username")
            .hexpand(true) // Make the entry expand to fill available space
            .build();
        username_row.append(&username_icon);
        username_row.append(&username_entry);

        // Create a horizontal box for password
        let password_row = GtkBox::builder()
            .orientation(gtk::Orientation::Horizontal)
            .spacing(10)
            .build();
        let password_icon = Image::from_icon_name("security-high-symbolic"); // Updated icon
        let password_entry = PasswordEntry::builder()
            .placeholder_text("Enter your password")
            .hexpand(true) // Make the entry expand to fill available space
            .build();
        password_row.append(&password_icon);
        password_row.append(&password_entry);

        // Create a login button
        let login_button = Button::builder()
            .label("Login")
            .margin_top(20) // Add margin above the button for better spacing
            .build();

        // Create a register button
        let register_button = Button::builder()
            .label("Register")
            .margin_top(5) // Add margin above the button for better spacing
            .build();

        // Create a "Start Quick Chat" button with a different style
        let quick_chat_button = Button::builder()
            .label("Start Quick Chat")
            .margin_top(20) // Add margin above the button for better spacing
            .build();
        quick_chat_button
            .style_context()
            .add_class("suggested-action"); // Apply a different style

        // Add rows and buttons to the login box
        login_box.append(&username_row);
        login_box.append(&password_row);
        login_box.append(&login_button);
        login_box.append(&register_button);
        login_box.append(&quick_chat_button);

        // Set login_box as the child of the login window
        login_window.set_child(Some(&login_box));

        // Connect the login button to handle login logic
        login_button.connect_clicked({
            let login_window = login_window.clone();
            let app = app.clone();
            let username_entry = username_entry.clone();
            let password_entry = password_entry.clone();
            move |_| {
                let username = username_entry.text().to_string();
                let password = password_entry.text().to_string();

                if username.is_empty() || password.is_empty() {
                    // Replace the error message with a user-friendly alert dialog
                    let dialog = gtk::MessageDialog::builder()
                        .transient_for(&login_window)
                        .modal(true)
                        .message_type(gtk::MessageType::Warning)
                        .buttons(gtk::ButtonsType::Ok)
                        .text("Warning")
                        .secondary_text("Username or password cannot be empty.")
                        .build();
                    dialog.run_async(|obj, _| {
                        obj.close();
                    });
                } else {
                    println!("Logging in with username: {}", username);
                    // Here you can add logic to verify the username and password

                    login_window.close();
                    build_ui(client)(&app);
                }
            }
        });

        // Connect the register button to handle registration logic
        register_button.connect_clicked({
            let login_window = login_window.clone();
            let app = app.clone();
            let username_entry = username_entry.clone();
            let password_entry = password_entry.clone();
            move |_| {
                let username = username_entry.text().to_string();
                let password = password_entry.text().to_string();

                if username.is_empty() || password.is_empty() {
                    // Replace the error message with a user-friendly alert dialog
                    let dialog = gtk::MessageDialog::builder()
                        .transient_for(&login_window)
                        .modal(true)
                        .message_type(gtk::MessageType::Warning)
                        .buttons(gtk::ButtonsType::Ok)
                        .text("Warning")
                        .secondary_text("Username or password cannot be empty.")
                        .build();
                    dialog.run_async(|obj, _| {
                        obj.close();
                    });
                } else {
                    println!("Register with username: {}", username);
                    // Here you can add logic to verify the username and password

                    login_window.close(); // Close the login window after successful login
                    build_ui(client)(&app);
                }
            }
        });

        // Connect the quick chat button to handle quick chat logic
        quick_chat_button.connect_clicked({
            let login_window = login_window.clone();
            let app = app.clone();
            move |_| {
                login_window.close(); // Close the login window after successful login
                build_ui(client)(&app);
            }
        });

        // Show the login window
        login_window.present();
    }
}

async fn response_text(
    buffer: gtk::TextBuffer,
    reply: impl Stream<Item = anyhow::Result<String>>,
    text_view: TextView,
) {
    let mut end = buffer.end_iter();
    let tag = buffer
        .create_tag(None, &[("foreground", &"green")])
        .unwrap();
    buffer.insert_with_tags(&mut end, "Assistant: ", &[&tag]);
    #[futures_async_stream::for_await]
    for c in reply {
        let text: String = c.unwrap();
        buffer.insert(&mut end, &text);
        // Scroll to end
        text_view.scroll_to_iter(&mut buffer.end_iter(), 0.0, false, 0.0, 0.0);
    }
    buffer.insert(&mut end, "\n");
    // Scroll to end
    text_view.scroll_to_iter(&mut buffer.end_iter(), 0.0, false, 0.0, 0.0);
}
