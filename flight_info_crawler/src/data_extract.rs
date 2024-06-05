use anyhow::Result;
use convert_case::{Case, Casing};
use futures::future::join_all;
use thirtyfour::prelude::*;

pub async fn get_table_headers(table_element: &WebElement) -> Result<Vec<String>> {
    let header_elements = table_element.query(By::Css("thead th")).any().await?;
    let headers = join_all(
        header_elements
            .iter()
            .map(|header_element| async { header_element.text().await.unwrap() }),
    )
    .await;

    Ok(headers)
}

fn remove_unit(text: &str) -> String {
    text.split_once(' ').unwrap().0.to_string()
}

pub async fn get_table_row(table_row_element: &WebElement) -> Result<Vec<String>> {
    let date = table_row_element
        .query(By::Css("th:nth-child(1)"))
        .single()
        .await?
        .text()
        .await?;

    let hour = table_row_element
        .query(By::Css("th:nth-child(2)"))
        .single()
        .await?
        .text()
        .await?;

    let weather_condition_code_class = table_row_element
        .query(By::Css("td:nth-child(3) > i"))
        .single()
        .await?
        .class_name()
        .await?
        .unwrap();
    let weather_condition_code = weather_condition_code_class
        .split_whitespace()
        .nth(1)
        .unwrap()
        .split_once('-')
        .unwrap()
        .1
        .to_case(Case::Title);

    let temperature = table_row_element
        .query(By::Css("td:nth-child(4)"))
        .single()
        .await?
        .text()
        .await?;
    let temperature = remove_unit(&temperature);

    let dew_point = table_row_element
        .query(By::Css("td:nth-child(5)"))
        .single()
        .await?
        .text()
        .await?;
    let dew_point = remove_unit(&dew_point);

    let sunshine_duration = table_row_element
        .query(By::Css("td:nth-child(6)"))
        .single()
        .await?
        .text()
        .await?;

    let total_recipitation = table_row_element
        .query(By::Css("td:nth-child(7)"))
        .single()
        .await?
        .text()
        .await?;
    let total_recipitation = remove_unit(&total_recipitation);

    let snow_depth = table_row_element
        .query(By::Css("td:nth-child(8)"))
        .single()
        .await?
        .text()
        .await?;

    let wind_direction_class = table_row_element
        .query(By::Css("td:nth-child(9) > i"))
        .single()
        .await?
        .class_name()
        .await?
        .unwrap();
    let wind_direction = wind_direction_class
        .split_whitespace()
        .nth(2)
        .unwrap()
        .split('-')
        .nth(1)
        .unwrap()
        .to_string();

    let wind_speed = table_row_element
        .query(By::Css("td:nth-child(10)"))
        .single()
        .await?
        .text()
        .await?;
    let wind_speed = remove_unit(&wind_speed);

    let peak_gust = table_row_element
        .query(By::Css("td:nth-child(11)"))
        .single()
        .await?
        .text()
        .await?;

    let air_pressure = table_row_element
        .query(By::Css("td:nth-child(12)"))
        .single()
        .await?
        .text()
        .await?;
    let air_pressure = remove_unit(&air_pressure);

    let relative_humidity = table_row_element
        .query(By::Css("td:nth-child(13)"))
        .single()
        .await?
        .text()
        .await?;
    let relative_humidity = remove_unit(&relative_humidity);

    Ok(vec![
        date,
        hour,
        weather_condition_code,
        temperature,
        dew_point,
        sunshine_duration,
        total_recipitation,
        snow_depth,
        wind_direction,
        wind_speed,
        peak_gust,
        air_pressure,
        relative_humidity,
    ])
}
