pub mod data_extract;

use std::time::Duration;

use anyhow::Result;
use chrono::{NaiveDate, TimeDelta};
use thirtyfour::prelude::*;
use tokio::{
    fs::{File, OpenOptions},
    io::AsyncWriteExt as _,
    time::sleep,
};

const FLIGHT_INFO_URL: &str = "https://meteostat.net/en/place/vn/thu-uc?s=48900";
const CSV_PATH: &str = "data.csv";
const DAY_COUNT: usize = 100;
const DAY_PER_PAGE: usize = 5;
const CLICK_TO_GET_FULL_DATA_PER_PAGE: usize = DAY_PER_PAGE * 24 / 10 - 1;

async fn write_row(csv_file: &mut File, row: &[String]) -> Result<()> {
    let row = row.join(",") + "\n";
    csv_file.write_all(row.as_bytes()).await?;

    Ok(())
}

#[tokio::main]
async fn main() -> Result<()> {
    // Init the selenium driver
    let capabilities = DesiredCapabilities::chrome();
    let driver = WebDriver::new("http://localhost:9515", capabilities).await?;

    // Latest record day
    let mut current_date = NaiveDate::from_ymd_opt(2024, 5, 29).unwrap();
    let time_delta = TimeDelta::days(4);

    // Open the csv file
    let mut csv_file = OpenOptions::new()
        .write(true)
        .create(true)
        .truncate(true)
        .open(CSV_PATH)
        .await?;
    let mut is_header_existed = false;

    for _ in 0..DAY_COUNT / DAY_PER_PAGE {
        // Goto the website containing flight info
        driver
            .goto(format!(
                "{}&t={}/{}",
                FLIGHT_INFO_URL,
                (current_date - time_delta).format("%Y-%m-%d"),
                current_date.format("%Y-%m-%d"),
            ))
            .await?;

        // Wait for website to load
        sleep(Duration::from_secs(10)).await;

        // Click the accept button
        let accept_button = driver
            .query(By::Css("div.modal-footer > button.btn.btn-primary"))
            .wait(Duration::from_secs(60), Duration::from_millis(100))
            .single()
            .await?;
        accept_button.click().await?;

        // Click button to show detailed
        let show_detailed_button = driver
            .query(By::Css("button.btn:nth-child(3)"))
            .single()
            .await?;
        show_detailed_button.click().await?;

        sleep(Duration::from_secs(10)).await;

        // Get the table
        let table_element = driver.query(By::Css(".table-bordered")).single().await?;

        // Sort table by date in descending order
        let date_header = table_element
            .query(By::Css("thead > tr > th:nth-child(1)"))
            .single()
            .await?;
        date_header.click().await?;
        date_header.click().await?;

        // Click show more until reach the end
        let show_more_button = driver.query(By::Css("button.ms-auto")).single().await?;
        for _ in 0..CLICK_TO_GET_FULL_DATA_PER_PAGE {
            show_more_button.click().await?;
        }

        if !is_header_existed {
            // Get table headers
            let table_headers = data_extract::get_table_headers(&table_element).await?;
            write_row(&mut csv_file, &table_headers).await?;
            is_header_existed = true;
        }

        // Get table rows
        let table_row_elements = table_element.query(By::Css("tbody > tr")).any().await?;
        for row_element in table_row_elements {
            let row = data_extract::get_table_row(&row_element).await?;
            write_row(&mut csv_file, &row).await?;
        }

        current_date -= time_delta;
        current_date -= TimeDelta::days(1);
    }

    Ok(())
}
