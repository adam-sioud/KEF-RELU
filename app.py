import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

# Load the dataset
@st.cache_data
def load_data(file):
    df = pd.read_csv(file)

    # Ensure 'date' is in datetime format
    df['date'] = pd.to_datetime(df['date'], errors='coerce')

    # Convert numeric columns, coercing errors to NaN
    numeric_columns = ['value', 'energy_prediction', 'anomaly', 'delta_energy', 'is_open', 'temp', 'cluster']
    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Drop rows with NaN in critical columns (excluding 'cluster')
    df.dropna(subset=['date', 'value', 'energy_prediction', 'anomaly', 'delta_energy'], inplace=True)

    return df

# Plot energy consumption and anomalies using Plotly
def plot_energy_data_with_anomalies(df, tagid, selected_clusters, selected_date_range):
    # Filter data for the selected TagID
    tagid_data = df[df['property_nr'] == tagid].copy()

    # Filter by selected date range
    start_date, end_date = selected_date_range
    tagid_data = tagid_data[(tagid_data['date'] >= pd.to_datetime(start_date)) & (tagid_data['date'] <= pd.to_datetime(end_date))]

    # Create a line plot for actual energy consumption
    fig = go.Figure()

    # Plot the value (actual energy consumption)
    fig.add_trace(
        go.Scatter(
            x=tagid_data['date'],
            y=tagid_data['value'],
            mode='lines',
            name='Actual Energy Consumption',
            line=dict(color='royalblue', width=2),
            hovertemplate='Date: %{x|%Y-%m-%d %H:%M}<br>Value: %{y:.2f}'
        )
    )

    # Overlay predicted energy consumption
    fig.add_trace(
        go.Scatter(
            x=tagid_data['date'],
            y=tagid_data['energy_prediction'],
            mode='lines',
            name='Predicted Energy Consumption',
            line=dict(color='green', dash='dash', width=2),
            opacity=0.6,
            hovertemplate='Date: %{x|%Y-%m-%d %H:%M}<br>Predicted: %{y:.2f}'
        )
    )

    # Filter anomalies for selected clusters
    if selected_clusters and "All" not in selected_clusters:
        anomalies = tagid_data[
            tagid_data['anomaly'] > 0.995
        ].copy()
        anomalies = anomalies[anomalies['cluster'].astype(str).isin(selected_clusters)]
    else:
        anomalies = tagid_data[tagid_data['anomaly'] > 0.995].copy()

    # Add anomaly points
    if not anomalies.empty:
        fig.add_trace(
            go.Scatter(
                x=anomalies['date'],
                y=anomalies['value'],
                mode='markers',
                name='Anomalies (> 0.99)',
                marker=dict(color='crimson', size=6, line=dict(width=1, color='darkred')),
                hovertemplate='Date: %{x|%Y-%m-%d %H:%M}<br>Anomaly Value: %{y:.2f}'
            )
        )

    # Convert clusters to strings for display
    cluster_str = ", ".join(map(str, selected_clusters)) if "All" not in selected_clusters else "All"

    # Update layout for better visuals
    fig.update_layout(
        title=f"Energy Consumption for TagID {tagid} (Clusters: {cluster_str})",
        xaxis_title="Date",
        yaxis_title="Energy Consumption",
        legend_title="Legend",
        template="plotly_white",
        height=600,
        font=dict(size=14),
        hovermode='x unified',
    )

    # Add a range slider to the x-axis
    fig.update_xaxes(rangeslider_visible=True)

    # Show the plot
    st.plotly_chart(fig, use_container_width=True)

# Streamlit app layout
def main():
    st.set_page_config(page_title="Energy Consumption Dashboard", layout="wide")

    # Sidebar content
    st.sidebar.title("🏢 Energy Consumption Dashboard")

    # Sidebar styling
    st.markdown(
        """
        <style>
        [data-testid="stSidebar"] {
            background-color: #f0f2f6;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # File upload in sidebar
    uploaded_file = st.sidebar.file_uploader("Upload your dataset CSV file", type="csv")
    if uploaded_file:
        try:
            df = load_data(uploaded_file)

            # Sidebar for selection
            st.sidebar.header("🔧 Filters")

            # Exclude NaNs and -99 in 'cluster' from cluster summaries and selection
            cluster_df = df.dropna(subset=['cluster']).copy()
            cluster_df = cluster_df[cluster_df['cluster'] != -99]

            # Calculate total delta_energy by TagID
            tagid_summary = (
                df.groupby('property_nr')
                .agg(total_delta_energy=('delta_energy', 'sum'))
                .reset_index()
                .sort_values(by='total_delta_energy', ascending=False)
            )

            # Create dropdown options with delta_energy values
            tagid_options = tagid_summary.apply(
                lambda row: f"{row['property_nr']} (Δ Energy: {row['total_delta_energy']:.2f})", axis=1
            )
            tagid_mapping = {f"{row['property_nr']} (Δ Energy: {row['total_delta_energy']:.2f})": row['property_nr'] for _, row in tagid_summary.iterrows()}

            # Select TagID with delta_energy displayed
            selected_tagid_display = st.sidebar.selectbox("Select a TagID (Building):", tagid_options)
            selected_tagid = tagid_mapping[selected_tagid_display]

            # Get all clusters for the selected TagID, excluding NaN and -99 clusters
            tagid_cluster_data = cluster_df[cluster_df['property_nr'] == selected_tagid].copy()

            cluster_summary = (
                tagid_cluster_data.groupby('cluster')
                .agg(
                    anomaly_count=('anomaly', lambda x: (x > 0.99).sum()),
                    delta_energy_sum=('delta_energy', 'sum')
                )
                .reset_index()
            )

            cluster_summary.rename(columns={'cluster': 'Cluster', 'anomaly_count': 'Anomaly Count', 'delta_energy_sum': 'Delta Energy (Sum)'}, inplace=True)

            st.sidebar.write("📊 Cluster Summary")
            st.sidebar.dataframe(cluster_summary, use_container_width=True)

            # Cluster selection
            cluster_options = ["All"] + cluster_summary['Cluster'].astype(str).tolist()
            selected_clusters = st.sidebar.multiselect(
                "Select Clusters (default: All):", cluster_options, default="All"
            )

            # Date range selection
            st.sidebar.subheader("📅 Date Range Selection")
            min_date = df['date'].min().date()
            max_date = df['date'].max().date()

            selected_date_range = st.sidebar.date_input(
                "Select Date Range",
                [min_date, max_date],
                min_value=min_date,
                max_value=max_date
            )

            # Plot data (includes all data, even if 'cluster' is NaN or -99)
            plot_energy_data_with_anomalies(df, selected_tagid, selected_clusters, selected_date_range)

            # Anomaly Analysis
            st.subheader("📈 Anomaly Analysis")

            # Filter data for selected tagid and date range (include all clusters)
            filtered_data = df[df['property_nr'] == selected_tagid].copy()
            start_date, end_date = selected_date_range
            filtered_data = filtered_data[(filtered_data['date'] >= pd.to_datetime(start_date)) & (filtered_data['date'] <= pd.to_datetime(end_date))]

            # For anomalies, filter based on selected clusters
            anomalies = filtered_data[filtered_data['anomaly'] > 0.99].copy()
            if "All" not in selected_clusters:
                # Exclude NaN and -99 clusters
                anomalies = anomalies.dropna(subset=['cluster'])
                anomalies = anomalies[anomalies['cluster'] != -99]
                anomalies = anomalies[anomalies['cluster'].astype(str).isin(selected_clusters)]

            if anomalies.empty:
                st.write("No anomalies detected for the selected TagID and clusters in the selected date range.")
            else:
                st.write("Detailed analysis of anomalies for the selected TagID and clusters.")

                # Create columns for side-by-side layout
                col1, col2 = st.columns(2)

                with col1:
                    # Top dates with most anomalies
                    anomalies_by_date = anomalies.groupby(anomalies['date'].dt.date).size().reset_index(name='Anomaly Count')
                    anomalies_by_date = anomalies_by_date.sort_values(by='Anomaly Count', ascending=False)

                    st.write("#### Top Dates with Most Anomalies")
                    st.dataframe(anomalies_by_date.head(10), use_container_width=True)

                    # Distribution by day of week
                    anomalies['day_of_week'] = anomalies['date'].dt.day_name()
                    anomalies_by_day = anomalies.groupby('day_of_week').size().reindex(
                        ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
                    ).reset_index(name='Anomaly Count')

                    st.write("#### Anomalies by Day of Week")
                    fig_day = px.bar(anomalies_by_day, x='day_of_week', y='Anomaly Count', title='Anomalies by Day of Week')
                    st.plotly_chart(fig_day, use_container_width=True)

                with col2:
                    # Distribution by hour
                    anomalies['hour'] = anomalies['date'].dt.hour
                    anomalies_by_hour = anomalies.groupby('hour').size().reset_index(name='Anomaly Count')

                    st.write("#### Anomalies by Hour")
                    fig_hour = px.bar(anomalies_by_hour, x='hour', y='Anomaly Count', title='Anomalies by Hour')
                    st.plotly_chart(fig_hour, use_container_width=True)

            # Display filtered data table below
            st.subheader("📝 Filtered Data")
            st.dataframe(filtered_data, use_container_width=True)

        except Exception as e:
            st.error(f"An error occurred: {e}")

    else:
        st.sidebar.info("👈 Please upload a CSV file to start.")

if __name__ == "__main__":
    main()
