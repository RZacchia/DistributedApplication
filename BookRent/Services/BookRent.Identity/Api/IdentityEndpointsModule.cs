namespace BookRent.Identity.Api;

internal static class IdentityEndpointsModule
{
    internal static void MapIdentityEndPoints(this IEndpointRouteBuilder app)
    {
        RouteGroupBuilder group = app.MapGroup("/identity");

        group.MapGet("/login", IdentityEndpoints.Login);
        group.MapPost("/register", IdentityEndpoints.Register);
        group.MapGet("/role/{id:guid}", IdentityEndpoints.GetUserRole);
    }
    
}