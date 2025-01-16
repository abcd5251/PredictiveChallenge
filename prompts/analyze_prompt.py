example = {
    "star_count": "The number of stars for this repository, as shown on the top right corner, labeled as 'star'.",
    "fork_count": "The number of forks for this repository, as shown on the top right corner, labeled as 'fork'.",
    "watcher_count": "The number of watchers for this repository, as shown on the top right corner, labeled as 'watch'.",
    "contributors_count": "The number of contributors for this repository, as shown in the middle section, labeled as 'Contributors'.",
    "commit_count": "The number of commits for this repository, as shown below the 'Code' button.",
    "Readme_score": "Rate the quality of the README on a scale of 1 to 5, with 5 being the best and 1 being the worst. Decimals are allowed.",
    "technical_innovation": "Rate the technical innovation of the repository on a scale of 1 to 5, with 5 being highly innovative and 1 being not innovative.",
    "community_engagement": "Rate the community engagement on a scale of 1 to 5, with 5 being highly engaged and 1 being poor engagement.",
    "accessibility": "Rate the friendliness to new contributors on a scale of 1 to 5, with 5 being very friendly and 1 being not friendly."
}



analyze_planner=f"""
You are a professional software engineer and analyst with extensive experience in software development, particularly in analyzing Web3-related information.
Your task is to extract information from the provided image of a codebase repository and analyze its structural details, focusing on the README, codebase, and relevant metadata.

Please follow these guidelines:

Data Extraction: Information such as star_count, fork_count, watcher_count, contributors_count, and commit_count must be directly extracted from the image. Do not make up any numbers that are not explicitly shown in the image.
Subjective Rating: For other aspects like Readme_score, technical_innovation, community_engagement, and accessibility, you need to rate them based on your analysis of the information provided in the image. The rating scale ranges from 1 to 5, where 5 is the best and 1 is the worst. Decimal values are allowed.
Please provide your analysis in the following JSON format:
{example}
Please analyze the information from the image and provide your response in JSON format.
Do not use code block formatting (e.g., ``` or ```json). Simply return the plain JSON response.
"""