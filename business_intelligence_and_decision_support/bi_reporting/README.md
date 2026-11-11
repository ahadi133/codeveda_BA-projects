# Social Media Sentiment Reporting

**Author:** Ahadi Nur

## Objective

Provide a self-service reporting asset for monitoring sentiment and engagement across social-media content.

## Dashboard Preview

![Social media sentiment dashboard preview](Dashboard%20Preview.png)

## Business Snapshot

- **710** cleaned posts analyzed.
- **58.9%** of posts are positive; **25.2%** negative; **15.9%** neutral.
- **30,517** likes and **15,300** retweets recorded.
- Average engagement: **64.5 per post**.
- The USA has the largest positive-post volume, while platform sentiment distributions are broadly similar.

## Included Files

- `social_media_sentiment_dashboard.pbix` — editable Power BI workbook for local presentation.
- `Dashboard Preview.png` — screenshot of the report dashboard.
- `hashtag_summary.csv` — ranked hashtag-level posts and engagement.

## Recommended Report Views

Use cards for post volume, positive share, and engagement; add sentiment distribution, time trend, country, platform, and top-hashtag visuals. Slicers for date, country, platform, and sentiment enable stakeholder exploration.

## Data Note

The PBIX relies on the cleaned sentiment CSV stored in `../../data_preparation_and_exploration/data_cleaning/`.
