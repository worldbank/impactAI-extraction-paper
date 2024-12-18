#!/bin/bash
echo "Entering main folder..." >> /home/agomberto/process_and_shutdown.log
# Change to the project directory
cd /home/agomberto/impactAI-extraction-paper

# Run the PDF processing script
echo "Starting PDF processing..." >> /home/agomberto/process_and_shutdown.log
poetry run python src/parse/parse_pdf.py --verbose >> /home/agomberto/process_and_shutdown.log 2>&1

# Check if the script ran successfully
if [ $? -eq 0 ]; then
    echo "Processing completed successfully. Shutting down VM..." >> /home/agomberto/process_and_shutdown.log
    sudo shutdown -h now
else
    echo "Processing encountered errors. Check logs for details." >> /home/agomberto/process_and_shutdown.log
fi
